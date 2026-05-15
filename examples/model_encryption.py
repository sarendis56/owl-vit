# SPDX-License-Identifier: Apache-2.0
"""
Secure encryption and benchmarking script for OWL-ViT using a PUF-emulated key.

Steps:
1) Obtain master key K from PUF emulator using OID and DID
2) Load OWL-ViT model
3) Encrypt first two transformer blocks' attention and FFN weights in-place
4) Run the existing batch benchmark pipeline on a dataset
5) Save results and an encrypted checkpoint

NOTE: This is a development script. The PUF is emulated via hardcoded CRPs.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch

# Ensure project root and examples are in path to import benchmark helpers
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import benchmark components
from examples.hf_benchmark import (
    BatchBenchmark,
    HFOwlViTPredictor,
    create_sample_dataset,
    extract_pascal_voc_dataset,
    download_pascal_voc_dataset,
    get_pascal_voc_prompts,
)  # type: ignore

# Import PUF and crypto utilities
from emulator import PUFEmulator  # type: ignore
from kdf import (  # type: ignore
    derive_layer_key,
    split_attn_ffn,
    subkey_to_arnold_params,
    subkey_to_perm_password,
)
from dual_encryption import DualEncryption  # type: ignore


def get_transformer_blocks(model) -> List[torch.nn.Module]:
    """
    Try to obtain the list of transformer blocks for OWL-ViT across common HF layouts.
    """
    # Common paths in HF models
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
    """
    Infer hidden_size and intermediate_size from a transformer block by inspecting FFN.
    """
    # Search for two Linear layers with complementary shapes
    for name, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            in_f = w.shape[1]
            out_f = w.shape[0]
            # Heuristic: FFN up-projection usually has out_f = intermediate, in_f = hidden
            if out_f > in_f and out_f % in_f == 0:
                return in_f, out_f
    # Fallback: try attention projection
    for name, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] == mod.weight.shape[1]:
            return mod.weight.shape[0], mod.weight.shape[1]
    raise RuntimeError("Could not infer hidden/intermediate sizes from block.")


def collect_attention_weights(block) -> dict:
    """
    Collect attention projection weights as a dict of tensors:
    keys: 'query','key','value','output'
    """
    attn = None
    # Find an attention submodule
    for name, mod in block.named_modules():
        lname = name.lower()
        if "attention" in lname and hasattr(mod, "__class__"):
            attn = mod
            break
    if attn is None:
        attn = block

    candidates = []
    for name, mod in attn.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.shape[0] == mod.weight.shape[1]:
            candidates.append((name.lower(), mod))

    # Try to map by name hints
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
    # output projection
    for key_hint in ["out", "o_proj", "proj", "output"]:
        for n, m in candidates:
            if key_hint in n:
                weights["output"] = m.weight.data
                break
        if "output" in weights:
            break

    # If mapping incomplete, fallback to first four square linears
    if len(weights) < 4 and len(candidates) >= 4:
        sq = [m.weight.data for _, m in candidates[:4]]
        weights = {"query": sq[0], "key": sq[1], "value": sq[2], "output": sq[3]}
    return weights


def collect_ffn_weights(block) -> dict:
    """
    Collect FFN weights as a dict with keys 'intermediate' and 'output'.
    - 'intermediate': usually shape (intermediate, hidden)
    - 'output': usually shape (hidden, intermediate)
    """
    inter = None
    out = None
    for name, mod in block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
            if w.shape[0] > w.shape[1] and inter is None:
                inter = w.data
            elif w.shape[0] < w.shape[1]:
                out = w.data
    if inter is None or out is None:
        # Fallback: scan again with name hints
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                lname = name.lower()
                if any(k in lname for k in ["intermediate", "fc1", "mlp.fc1", "dense1"]) and inter is None:
                    inter = mod.weight.data
                if any(k in lname for k in ["output", "fc2", "mlp.fc2", "dense2"]) and out is None:
                    out = mod.weight.data
    return {"intermediate": inter, "output": out}


def encrypt_first_two_layers_inplace(model, master_key: bytes, device: str = "cuda"):
    blocks = get_transformer_blocks(model)
    num_layers = min(2, len(blocks))
    for layer_index in range(num_layers):
        block = blocks[layer_index]
        hidden_size, intermediate_size = infer_hidden_and_intermediate_sizes(block)

        layer_key = derive_layer_key(master_key, layer_index)
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

        # Collect weights
        attn_w = collect_attention_weights(block)
        ffn_w = collect_ffn_weights(block)

        # Encrypt attention
        enc_attn = de.encrypt_attention_weights(attn_w)
        # Write back
        for k, v in enc_attn.items():
            attn_w[k].copy_(v)

        # Encrypt FFN (use perm matrix idx 0)
        enc_ffn = de.encrypt_ffn_weights(ffn_w, permutation_matrix_idx=0)
        if enc_ffn.get("intermediate") is not None and ffn_w.get("intermediate") is not None:
            ffn_w["intermediate"].copy_(enc_ffn["intermediate"]) 
        if enc_ffn.get("output") is not None and ffn_w.get("output") is not None:
            ffn_w["output"].copy_(enc_ffn["output"]) 


def main():
    parser = argparse.ArgumentParser(description="Encrypt first two layers of OWL-ViT and benchmark")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory or annotation JSON file")
    parser.add_argument("--oid", type=str, default="OID-ALPHA", help="Owner ID for PUF challenge")
    parser.add_argument("--did", type=str, default="DID-0001-DEMO", help="Device ID for PUF lookup")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32", help="Model name")
    parser.add_argument("--prompts", type=str, default="", help="Detection prompts (comma-separated); ignored when --use_pascal_voc_prompts")
    parser.add_argument("--output_dir", type=str, default="./secure_benchmark_results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="./secure_benchmark_results/encrypted_checkpoint.pt", help="Path to save encrypted checkpoint")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection threshold")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images")
    parser.add_argument("--warmup_runs", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--viz_threshold", type=float, default=0.5, help="Visualization threshold")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization images")
    parser.add_argument("--use_channels_last", action="store_true", help="Use channels_last for GPU")
    parser.add_argument("--create_sample_dataset", action="store_true", help="Create a small sample dataset for testing")
    parser.add_argument("--download_pascal_voc", action="store_true", help="Download Pascal VOC 2012 dataset")
    parser.add_argument("--extract_pascal_voc", action="store_true", help="Extract Pascal VOC 2012 dataset from data/pascal.zip (relative to the current working directory)")
    parser.add_argument("--use_pascal_voc_prompts", action="store_true", default=True, help="Use Pascal VOC class names as prompts (default: True)")
    parser.add_argument("--coco_eval", action="store_true", default=True, help="Compute COCO-style metrics with pycocotools (default: True)")
    parser.add_argument("--eval_threshold", type=float, default=0.2, help="Score threshold for reporting precision/recall/F1 (does not affect AP)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) PUF master key
    puf = PUFEmulator()
    master_key = puf.get_master_key(args.oid, args.did)
    if master_key is None:
        print("PUF could not derive master key for provided OID/DID; aborting.")
        sys.exit(1)

    # 2) Load OWL-ViT via predictor
    predictor = HFOwlViTPredictor(
        model_name=args.model,
        device=device,
        quantization="fp16",
        use_channels_last=args.use_channels_last,
    )

    model = predictor.model
    model.eval()

    # 3) Encrypt first two transformer blocks in-place
    encrypt_first_two_layers_inplace(model, master_key, device=device)

    # Prepare dataset path similar to hf_benchmark
    if args.extract_pascal_voc:
        print("Extracting Pascal VOC 2012 dataset from data/pascal.zip...")
        dataset_path = extract_pascal_voc_dataset("data/pascal.zip", args.dataset, args.max_images)
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
        # Auto-detect Pascal VOC layout and create annotations if needed
        dataset_path = args.dataset
        ds_path = Path(dataset_path)
        if ds_path.is_dir():
            ann_dir = ds_path / "Annotations"
            img_dir = ds_path / "JPEGImages"
            if ann_dir.exists() and img_dir.exists():
                try:
                    from examples.hf_benchmark import create_pascal_voc_annotations  # lazy import
                    annotations_file = ds_path / "pascal_voc_annotations.json"
                    if not annotations_file.exists():
                        num = create_pascal_voc_annotations(str(ds_path), str(annotations_file), args.max_images)
                        if num == 0:
                            print("No annotations created from VOC directory; exiting.")
                            sys.exit(1)
                    dataset_path = str(annotations_file)
                except Exception as e:
                    print(f"Failed to auto-create VOC annotations: {e}")
                    # fall back to directory loading (performance only)

    # 4) Run benchmark with our predictor/model
    # Create a benchmark instance and override its predictor with our modified one
    bench = BatchBenchmark(
        model_name=args.model,
        device=device,
        quantization="fp16",
        use_channels_last=args.use_channels_last,
    )
    bench.predictor = predictor

    # Parse prompts like hf_benchmark (default to Pascal VOC prompts)
    if args.use_pascal_voc_prompts:
        prompts = get_pascal_voc_prompts()
        print(f"Using Pascal VOC prompts: {len(prompts)} classes")
    else:
        if not args.prompts:
            print("No prompts provided and --use_pascal_voc_prompts disabled; exiting.")
            sys.exit(1)
        prompts = args.prompts.strip("[]()").split(',')
        prompts = [p.strip().strip('\"\'') for p in prompts]

    os.makedirs(args.output_dir, exist_ok=True)

    # Guard against empty dataset before calling into warmup in run_benchmark
    try:
        image_paths, _ = bench.load_dataset(dataset_path)
    except Exception:
        image_paths = []
    if len(image_paths) == 0:
        print("No images found for the provided dataset path. Please use --extract_pascal_voc or provide an annotations JSON.")
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
        batch_size=args.batch_size,
        use_pascal_voc_prompts=args.use_pascal_voc_prompts,
    )

    bench.print_results(results)

    # Save results JSON like the baseline
    results_file = os.path.join(args.output_dir, "secure_benchmark_results.json")
    bench.save_results(results, results_file)

    # 5) Save encrypted checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "info": {
        "model_name": args.model,
        "encrypted_layers": 2,
        "device": device,
    }}, str(ckpt_path))
    print(f"Encrypted checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
