# SPDX-License-Identifier: Apache-2.0
"""
Real-time secure benchmark: per-batch decrypt → infer → re-encrypt.

- Loads OWL-ViT (optionally encrypts N layers at start to simulate encrypted-at-rest)
- For each batch in evaluation:
  - Decrypt the specified layers on-the-fly
  - Run inference
  - Re-encrypt immediately before moving to next batch
- Reports detection accuracy (should match baseline) and FPS including crypto overhead
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hf_benchmark import (
    BatchBenchmark,
    HFOwlViTPredictor,
    extract_pascal_voc_dataset,
    download_pascal_voc_dataset,
    create_sample_dataset,
    get_pascal_voc_prompts,
)  # type: ignore

from emulator import PUFEmulator  # type: ignore
from kdf import (
    derive_layer_key,
    split_attn_ffn,
    subkey_to_arnold_params,
    subkey_to_perm_password,
)  # type: ignore
from dual_encryption import DualEncryption  # type: ignore


def get_transformer_blocks(model) -> List[torch.nn.Module]:
    candidates = [
        ["owlvit", "vision_model", "encoder", "layers"],
        ["owlvit", "vision_model", "encoder", "layer"],
        ["vit", "encoder", "layer"],
        ["vision_model", "encoder", "layers"],
        ["vision_model", "encoder", "layer"],
    ]
    for path in candidates:
        node = model
        ok = True
        for attr in path:
            if hasattr(node, attr):
                node = getattr(node, attr)
            else:
                ok = False
                break
        if ok and isinstance(node, (list, tuple)):
            return list(node)
        if ok and hasattr(node, "__len__"):
            try:
                return [node[i] for i in range(len(node))]
            except Exception:
                pass
    raise RuntimeError("Could not locate transformer blocks in model.")


def infer_hidden_and_intermediate_sizes(block) -> Tuple[int, int]:
    for _, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            in_f = w.shape[1]
            out_f = w.shape[0]
            if out_f > in_f and out_f % in_f == 0:
                return in_f, out_f
    for _, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] == mod.weight.shape[1]:
            return mod.weight.shape[0], mod.weight.shape[1]
    raise RuntimeError("Could not infer hidden/intermediate sizes from block.")


def collect_attention_weights(block) -> dict:
    attn = None
    for name, mod in block.named_modules():
        if "attention" in name.lower():
            attn = mod
            break
    if attn is None:
        attn = block
    candidates = []
    for name, mod in attn.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] == mod.weight.shape[1]:
            candidates.append((name.lower(), mod))
    weights = {}
    for key_hint in ["query", "q_proj", "q"]:
        for n, m in candidates:
            if key_hint in n:
                weights["query"] = m.weight.data
                break
        if "query" in weights:
            break
    for key_hint in ["key", "k_proj", "k"]:
        for n, m in candidates:
            if key_hint in n:
                weights["key"] = m.weight.data
                break
        if "key" in weights:
            break
    for key_hint in ["value", "v_proj", "v"]:
        for n, m in candidates:
            if key_hint in n:
                weights["value"] = m.weight.data
                break
        if "value" in weights:
            break
    for key_hint in ["out", "o_proj", "proj", "output"]:
        for n, m in candidates:
            if key_hint in n:
                weights["output"] = m.weight.data
                break
        if "output" in weights:
            break
    if len(weights) < 4 and len(candidates) >= 4:
        sq = [m.weight.data for _, m in candidates[:4]]
        weights = {"query": sq[0], "key": sq[1], "value": sq[2], "output": sq[3]}
    return weights


def collect_ffn_weights(block) -> dict:
    inter = None
    out = None
    for _, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            if w.shape[0] > w.shape[1] and inter is None:
                inter = w.data
            elif w.shape[0] < w.shape[1]:
                out = w.data
    return {"intermediate": inter, "output": out}


class DecryptOnTheFlyPredictor(HFOwlViTPredictor):
    def __init__(self, master_key: bytes, num_layers: int, device: str = "cuda", **kwargs):
        super().__init__(device=device, **kwargs)
        self.master_key = master_key
        self.num_layers = num_layers

    def _apply_crypto(self, decrypt: bool):
        model = self.model
        device = self.device
        blocks = get_transformer_blocks(model)
        num_layers = min(self.num_layers, len(blocks))
        for layer_index in range(num_layers):
            block = blocks[layer_index]
            hidden_size, _ = infer_hidden_and_intermediate_sizes(block)
            layer_key = derive_layer_key(self.master_key, layer_index)
            attn_subkey, ffn_subkey = split_attn_ffn(layer_key)
            arnold_key = subkey_to_arnold_params(attn_subkey, hidden_size)
            perm_password = subkey_to_perm_password(ffn_subkey)
            de = DualEncryption(
                arnold_key=arnold_key,
                password=perm_password,
                num_permutation_matrices=1,
                matrix_size=hidden_size,
                device=device,
                dtype=torch.float32,
            )
            attn_w = collect_attention_weights(block)
            ffn_w = collect_ffn_weights(block)
            if decrypt:
                # reverse operations
                dec_attn = de.decrypt_attention_weights(attn_w)
                for k, v in dec_attn.items():
                    attn_w[k].copy_(v)
                dec_ffn = de.decrypt_ffn_weights(ffn_w, permutation_matrix_idx=0)
                if dec_ffn.get("intermediate") is not None and ffn_w.get("intermediate") is not None:
                    ffn_w["intermediate"].copy_(dec_ffn["intermediate"]) 
                if dec_ffn.get("output") is not None and ffn_w.get("output") is not None:
                    ffn_w["output"].copy_(dec_ffn["output"]) 
            else:
                enc_attn = de.encrypt_attention_weights(attn_w)
                for k, v in enc_attn.items():
                    attn_w[k].copy_(v)
                enc_ffn = de.encrypt_ffn_weights(ffn_w, permutation_matrix_idx=0)
                if enc_ffn.get("intermediate") is not None and ffn_w.get("intermediate") is not None:
                    ffn_w["intermediate"].copy_(enc_ffn["intermediate"]) 
                if enc_ffn.get("output") is not None and ffn_w.get("output") is not None:
                    ffn_w["output"].copy_(enc_ffn["output"]) 

    def predict_batch(self, images: List, text: List[str], threshold: float = 0.0):
        # Decrypt → infer → re-encrypt
        self._apply_crypto(decrypt=True)
        try:
            return super().predict_batch(images, text, threshold)
        finally:
            self._apply_crypto(decrypt=False)


def main():
    parser = argparse.ArgumentParser(description="Real-time secure benchmark: per-batch decrypt/infer/re-encrypt")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory or annotation JSON file")
    parser.add_argument("--oid", type=str, default="OID-ALPHA", help="Owner ID for PUF challenge")
    parser.add_argument("--did", type=str, default="DID-0001-DEMO", help="Device ID for PUF lookup")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./secure_inference_benchmark_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images")
    parser.add_argument("--warmup_runs", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--viz_threshold", type=float, default=0.5, help="Visualization threshold")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    parser.add_argument("--use_channels_last", action="store_true", help="Use channels_last for GPU")
    parser.add_argument("--extract_pascal_voc", action="store_true", help="Extract Pascal VOC 2012 dataset from /data/pascal.zip")
    parser.add_argument("--download_pascal_voc", action="store_true", help="Download Pascal VOC 2012 dataset")
    parser.add_argument("--create_sample_dataset", action="store_true", help="Create sample dataset")
    parser.add_argument("--use_pascal_voc_prompts", action="store_true", default=True, help="Use Pascal VOC prompts")
    parser.add_argument("--coco_eval", action="store_true", default=True, help="Compute COCO metrics")
    parser.add_argument("--eval_threshold", type=float, default=0.2, help="Operating point metrics threshold")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of initial layers to protect")
    parser.add_argument("--assume_encrypted", action="store_true", help="Assume model is already encrypted at rest")
    parser.add_argument("--quantization", type=str, default="none", 
                        choices=["none", "fp16", "cpu-int8"],
                        help="Quantization: none(fp32 GPU) | fp16(GPU) | cpu-int8(dynamic int8 on CPU)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PUF master key
    puf = PUFEmulator()
    master_key = puf.get_master_key(args.oid, args.did)
    if master_key is None:
        print("PUF could not derive master key; exiting.")
        sys.exit(1)

    # Dataset prep
    if args.extract_pascal_voc:
        print("Extracting Pascal VOC 2012 dataset from /data/pascal.zip...")
        dataset_path = extract_pascal_voc_dataset("/data/pascal.zip", args.dataset, args.max_images)
        if dataset_path is None:
            print("Failed to extract Pascal VOC dataset. Exiting.")
            sys.exit(1)
    elif args.download_pascal_voc:
        print("Downloading Pascal VOC 2012 dataset...")
        dataset_path = download_pascal_voc_dataset(args.dataset, args.max_images)
        if dataset_path is None:
            print("Failed to download Pascal VOC dataset. Exiting.")
            sys.exit(1)
    elif args.create_sample_dataset:
        dataset_path = create_sample_dataset(args.dataset, 10)
    else:
        dataset_path = args.dataset

    # Predictor with on-the-fly crypto
    # Select device by quantization mode
    if args.quantization == "cpu-int8":
        device = "cpu"
    else:
        device = device

    predictor = DecryptOnTheFlyPredictor(
        master_key=master_key,
        num_layers=args.num_layers,
        device=device,
        model_name=args.model,
        quantization=args.quantization,
        use_channels_last=args.use_channels_last if device == "cuda" and args.quantization == "fp16" else False,
    )

    model = predictor.model
    model.eval()

    # If not already encrypted, encrypt at-rest now so decrypt step is meaningful
    if not args.assume_encrypted:
        predictor._apply_crypto(decrypt=False)

    bench = BatchBenchmark(
        model_name=args.model,
        device=device,
        quantization=args.quantization,
        use_channels_last=args.use_channels_last if device == "cuda" and args.quantization == "fp16" else False,
    )
    bench.predictor = predictor

    # Prompts
    if args.use_pascal_voc_prompts:
        prompts = get_pascal_voc_prompts()
        print(f"Using Pascal VOC prompts: {len(prompts)} classes")
    else:
        print("Please provide prompts via --prompts (not implemented in this script)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure dataset not empty
    try:
        image_paths, _ = bench.load_dataset(dataset_path)
    except Exception:
        image_paths = []
    if len(image_paths) == 0:
        print("No images found for dataset; exiting.")
        sys.exit(1)

    results = bench.run_benchmark(
        dataset_path=dataset_path,
        prompts=prompts,
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_visualizations=args.save_visualizations,
        max_images=args.max_images,
        warmup_runs=args.warmup_runs,
        viz_threshold=args.viz_threshold,
        coco_eval=args.coco_eval,
        eval_threshold=args.eval_threshold,
        batch_size=args.batch,
        use_pascal_voc_prompts=args.use_pascal_voc_prompts,
    )

    bench.print_results(results)
    results_file = os.path.join(args.output_dir, "secure_inference_benchmark_results.json")
    bench.save_results(results, results_file)
    print(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
