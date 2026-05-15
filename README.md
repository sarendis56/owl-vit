<h1 align="center">NanoOWL</h1>

<p align="center"><a href="#usage"/>👍 Usage</a> - <a href="#performance"/>⏱️ Performance</a> - <a href="#setup">🛠️ Setup</a> - <a href="#examples">🤸 Examples</a> <br> - <a href="#acknowledgement">👏 Acknowledgment</a> - <a href="#see-also">🔗 See also</a></p>

NanoOWL is a project that optimizes [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) to run 🔥 ***real-time*** 🔥 on [NVIDIA Jetson Orin Platforms](https://store.nvidia.com/en-us/jetson/store) with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).  NanoOWL also introduces a new "tree detection" pipeline that combines OWL-ViT and CLIP to enable nested detection and classification of anything, at any level, simply by providing text.

<p align="center">
<img src="assets/jetson_person_2x.gif" height="50%" width="50%"/></p>

> Interested in detecting object masks as well?  Try combining NanoOWL with
> [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) for zero-shot open-vocabulary 
> instance segmentation.

<a id="usage"></a>
## 👍 Usage

You can use NanoOWL in Python like this

```python3
from nanoowl.owl_predictor import OwlPredictor

predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    image_encoder_engine="data/owlvit-base-patch32-image-encoder.engine"
)

image = PIL.Image.open("assets/owl_glove_small.jpg")

output = predictor.predict(image=image, text=["an owl", "a glove"], threshold=0.1)

print(output)
```

Or better yet, to use OWL-ViT in conjunction with CLIP to detect and classify anything,
at any level, check out the tree predictor example below!

> See [Setup](#setup) for instructions on how to build the image encoder engine.

<a id="performance"></a>
## ⏱️ Performance

NanoOWL runs real-time on Jetson Orin Nano.

<table style="border-top: solid 1px; border-left: solid 1px; border-right: solid 1px; border-bottom: solid 1px">
    <thead>
        <tr>
            <th rowspan=1 style="text-align: center; border-right: solid 1px">Model †</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">Image Size</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">Patch Size</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">⏱️ Jetson Orin Nano (FPS)</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">⏱️ Jetson AGX Orin (FPS)</th>
            <th colspan=1 style="text-align: center; border-right: solid 1px">🎯 Accuracy (mAP)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align: center; border-right: solid 1px">OWL-ViT (ViT-B/32)</td>
            <td style="text-align: center; border-right: solid 1px">768</td>
            <td style="text-align: center; border-right: solid 1px">32</td>
            <td style="text-align: center; border-right: solid 1px">TBD</td>
            <td style="text-align: center; border-right: solid 1px">95</td>
            <td style="text-align: center; border-right: solid 1px">28</td>
        </tr>
        <tr>
            <td style="text-align: center; border-right: solid 1px">OWL-ViT (ViT-B/16)</td>
            <td style="text-align: center; border-right: solid 1px">768</td>
            <td style="text-align: center; border-right: solid 1px">16</td>
            <td style="text-align: center; border-right: solid 1px">TBD</td>
            <td style="text-align: center; border-right: solid 1px">25</td>
            <td style="text-align: center; border-right: solid 1px">31.7</td>
        </tr>
    </tbody>
</table>

<a id="setup"></a>
## 🛠️ Setup

1. Install the dependencies

    1. Install PyTorch

    2. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
    3. Install NVIDIA TensorRT
    4. Install the Transformers library

        ```bash
        python3 -m pip install transformers
        ```
    5. (optional) Install NanoSAM (for the instance segmentation example)

2. Install the NanoOWL package.

    ```bash
    git clone https://github.com/NVIDIA-AI-IOT/nanoowl
    cd nanoowl
    python3 setup.py develop --user
    ```

3. Build the TensorRT engine for the OWL-ViT vision encoder

    ```bash
    mkdir -p data
    python3 -m nanoowl.build_image_encoder_engine \
        data/owl_image_encoder_patch32.engine
    ```
    

4. Run an example prediction to ensure everything is working

    ```bash
    cd examples
    python3 owl_predict.py \
        --prompt="[an owl, a glove]" \
        --threshold=0.1 \
        --image_encoder_engine=../data/owl_image_encoder_patch32.engine
    ```

That's it!  If everything is working properly, you should see a visualization saved to ``data/owl_predict_out.jpg``.  

<a id="examples"></a>
## 🤸 Examples

### Example 1 - Basic prediction

<img src="assets/owl_predict_out.jpg" height="256px"/>

This example demonstrates how to use the TensorRT optimized OWL-ViT model to
detect objects by providing text descriptions of the object labels.

To run the example, first navigate to the examples folder

```bash
cd examples
```

Then run the example

```bash
python3 owl_predict.py \
    --prompt="[an owl, a glove]" \
    --threshold=0.1 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

By default the output will be saved to ``data/owl_predict_out.jpg``. 

You can also use this example to profile inference.  Simply set the flag ``--profile``.

### Example 2 - Tree prediction

<img src="assets/tree_predict_out.jpg" height="256px"/>

This example demonstrates how to use the tree predictor class to detect and
classify objects at any level.

To run the example, first navigate to the examples folder

```bash
cd examples
```

To detect all owls, and the detect all wings and eyes in each detect owl region
of interest, type

```bash
python3 tree_predict.py \
    --prompt="[an owl [a wing, an eye]]" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

By default the output will be saved to ``data/tree_predict_out.jpg``.

To classify the image as indoors or outdoors, type

```bash
python3 tree_predict.py \
    --prompt="(indoors, outdoors)" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```

To classify the image as indoors or outdoors, and if it's outdoors then detect
all owls, type

```bash
python3 tree_predict.py \
    --prompt="(indoors, outdoors [an owl])" \
    --threshold=0.15 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine
```


### Example 3 - Tree prediction (Live Camera)

<img src="assets/jetson_person_2x.gif" height="50%" width="50%"/>

This example demonstrates the tree predictor running on a live camera feed with
live-edited text prompts.  To run the example

1. Ensure you have a camera device connected

2. Launch the demo
    ```bash
    cd examples/tree_demo
    python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
    ```
3. Second, open your browser to ``http://<ip address>:7860``
4. Type whatever prompt you like to see what works!  Here are some examples
    - Example: [a face [a nose, an eye, a mouth]]
    - Example: [a face (interested, yawning / bored)]
    - Example: (indoors, outdoors)

## ISCAS 2026 Demo Setup and Evaluation Guide

This section describes how to set up the environment from scratch on commodity GPU hardware (e.g., NVIDIA A100) and run the three benchmark scripts that do **not** require the TensorRT-accelerated NanoOWL path. These three scripts are sufficient to reproduce the paper's Table 1 numbers (baseline accuracy/FPS, encrypted-model failure, and authorized on-the-fly decrypt-infer-re-encrypt overhead).

### 1. Prerequisites

- An NVIDIA GPU with CUDA support (tested on Jetson Orin Nano with JetPack 6 / CUDA 12.6, and on x86 hosts with CUDA 12.x).
- Docker with NVIDIA Container Toolkit, or a host Python 3.10+ environment.
- The Pascal VOC 2012 archive available at `/data/pascal.zip` inside the runtime environment. The benchmark scripts extract and convert annotations automatically when given `--extract_pascal_voc`.

### 2. Environment Setup

#### Option A: Docker (recommended)

From the repository root, build and start the container:

```bash
docker build -t nanoowl:23-01 -f docker/23-01/Dockerfile docker/23-01
docker run -it --rm --gpus all --ipc host --shm-size 14G \
    -v $(pwd):/nanoowl \
    -v /path/to/pascal.zip:/data/pascal.zip \
    nanoowl:23-01
```

Inside the container:

```bash
cd /nanoowl
python3 setup.py develop --user
```

#### Option B: Bare-metal Python

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers timm accelerate pillow numpy tqdm matplotlib opencv-python pycocotools
git clone https://github.com/openai/CLIP.git && python3 -m pip install ./CLIP
git clone <this-repo> nanoowl && cd nanoowl
python3 setup.py develop --user
```

The three scripts used below depend only on `torch`, `transformers`, `pillow`, `numpy`, `tqdm`, `matplotlib`, and (optionally) `pycocotools`. They do **not** require `torchvision`, `tensorrt`, or `torch2trt`.

### 3. Dataset Preparation

The first invocation of any script with `--extract_pascal_voc` extracts `/data/pascal.zip` into `./pascal_voc_extracted/` and builds a `pascal_voc_annotations.json` file. Subsequent runs detect the existing annotations and skip re-extraction, so the flag can be left on for convenience.

### 4. Running the Benchmarks

All commands are run from the `examples/` directory:

```bash
cd examples
```

#### 4.1 Baseline (unprotected model)

Produces the **Baseline** column of Table 1 — mAP@0.5 ≈ 0.61 and ~10.69 FPS on Jetson Orin Nano with batch size 8 and fp16.

```bash
python3 hf_benchmark.py \
  --extract_pascal_voc \
  --dataset ./pascal_voc_extracted \
  --use_pascal_voc_prompts \
  --max_images 100 \
  --batch_size 8 \
  --quantization fp16 \
  --use_channels_last \
  --threshold 0.1 \
  --viz_threshold 0.2 \
  --eval_threshold 0.2 \
  --output_dir ./benchmark_results \
  --save_visualizations \
  --coco_eval
```

Output:

- Console: per-class AP, COCO mAP, FPS, latency breakdown.
- `./benchmark_results/result_*.jpg`: annotated detection visualizations.
- `./benchmark_results/benchmark_results.json`: structured metrics.

#### 4.2 Unauthorized Device (statically encrypted model)

Simulates an attacker running the encrypted weights without the correct PUF-derived key. Produces the **Enc. 1 Layer / Enc. 2 Layers** mAP collapse (≈ 0.00) in Table 1. The script encrypts the first two transformer layers in place using a key derived from a default PUF Owner/Device ID, then runs inference on the encrypted model.

```bash
python3 model_encryption.py \
  --extract_pascal_voc \
  --dataset ./pascal_voc_extracted \
  --max_images 100 \
  --batch_size 8 \
  --viz_threshold 0.05 \
  --output_dir ./secure_benchmark_results \
  --save_visualizations \
  --coco_eval
```

The visualizations in `./secure_benchmark_results/result_*.jpg` will show either no detections or visibly wrong boxes, illustrating the catastrophic failure on unauthorized hardware.

#### 4.3 Authorized Device (per-batch decrypt-infer-re-encrypt)

Produces the FPS rows for **Enc. 1 Layer** (8.58 FPS) and **Enc. 2 Layers** (7.41 FPS) of Table 1 while restoring full Baseline accuracy. The model is held in encrypted form at rest, and decrypted into plaintext only for the duration of each forward pass.

Two protected layers (matches the paper's Enc. 2 Layers column):

```bash
python3 secure_inference_benchmark.py \
  --extract_pascal_voc \
  --dataset ./pascal_voc_extracted \
  --use_pascal_voc_prompts \
  --max_images 100 \
  --batch 8 \
  --num_layers 2 \
  --quantization fp16 \
  --threshold 0.1 \
  --viz_threshold 0.2 \
  --eval_threshold 0.2 \
  --output_dir ./secure_inference_benchmark_results \
  --save_visualizations \
  --coco_eval
```

For the **Enc. 1 Layer** row, change `--num_layers 2` to `--num_layers 1` and re-run.

### 5. Expected Results (Pascal VOC, batch 8, fp16, 100 images)

| Script                                         | Layers Encrypted | mAP@0.5 | FPS (Jetson Orin Nano) |
| ---------------------------------------------- | ---------------- | ------- | ---------------------- |
| `hf_benchmark.py`                              | 0 (Baseline)     | ~0.61   | ~10.69                 |
| `model_encryption.py`                          | 2 (Unauthorized) | ~0.00   | n/a                    |
| `secure_inference_benchmark.py --num_layers 1` | 1 (Authorized)   | ~0.61   | ~8.58                  |
| `secure_inference_benchmark.py --num_layers 2` | 2 (Authorized)   | ~0.61   | ~7.41                  |

Absolute FPS will differ on other hardware (substantially higher on A100, for instance), but the relative overhead pattern between the rows should hold.

### 6. Notes

- **Skipped script:** `rt_benchmark.py` uses the NanoOWL TensorRT-accelerated path via `nanoowl.owl_predictor.OwlPredictor`, which imports `torchvision.ops.roi_align` at module load time and requires a pre-built `.engine` file. It is not needed to reproduce any number in the paper and is intentionally omitted from this guide.
- **PUF emulation:** `examples/emulator.py` ships with hardcoded challenge-response pairs for development. The default `OID-ALPHA` / `DID-0001-DEMO` are valid and will derive a usable master key. For a real deployment, replace the emulator with the Ultra96-V2 Arbiter PUF interface described in Section II of the paper.
- **Disk usage:** `--save_visualizations` writes one JPEG per processed image. For the on-site live demonstration, these visualizations are what visitors see — no camera is needed.

<a id="acknowledgement"></a>
## 👏 Acknowledgement

Thanks to the authors of [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) for the great open-vocabluary detection work.

<a id="see-also"></a>
## 🔗 See also

- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) - A real-time Segment Anything (SAM) model variant for NVIDIA Jetson Orin platforms.
- [Jetson Introduction to Knowledge Distillation Tutorial](https://github.com/NVIDIA-AI-IOT/jetson-intro-to-distillation) - For an introduction to knowledge distillation as a model optimization technique.
- [Jetson Generative AI Playground](https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/) - For instructions and tips for using a variety of LLMs and transformers on Jetson.
- [Jetson Containers](https://github.com/dusty-nv/jetson-containers) - For a variety of easily deployable and modular Jetson Containers
