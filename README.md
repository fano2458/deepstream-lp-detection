# DeepStream-Yolo

A license plate detection pipeline that trains a YOLO26 model and deploys it with NVIDIA DeepStream for real-time, multi-stream inference on GPU hardware (tested on Jetson).

The pipeline covers the full workflow: dataset download, hyperparameter search, model training, TensorRT export, and DeepStream integration — with benchmarking tools to measure throughput and hardware utilization.

---

## How it works

### 1. Dataset

The dataset is sourced from Roboflow (`license-plate-recognition-rxg4e`, version 12) in YOLO format and downloaded via [test_quantization/download_data.py](test_quantization/download_data.py).

### 2. Hyperparameter search

[test_quantization/hyper_parameters_search.ipynb](test_quantization/hyper_parameters_search.ipynb) runs a lightweight tuning sweep on `yolo26n.pt` to find good learning rate, momentum, weight decay, and augmentation settings before the full training run.

### 3. Training

[test_quantization/train.ipynb](test_quantization/train.ipynb) trains the model with the best hyperparameters:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="/home/nailf/test_quantization/data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=128,
    optimizer="AdamW",
    flipud=0.0,
    close_mosaic=10,
    cos_lr=True,
    patience=5,
    device=0,
    amp=True,
    workers=8,
    cache=True,
)
```

Training stopped early at epoch 37. Best results:

| Metric | Value |
|---|---|
| mAP50 | 0.969 |
| mAP50-95 | 0.691 |
| Precision | 0.972 |
| Recall | 0.932 |

### 4. Validation and TensorRT export

[test_quantization/validate.ipynb](test_quantization/validate.ipynb) validates `best.pt`, exports a TensorRT engine, and re-validates the engine to confirm accuracy is preserved.

### 5. Export to DeepStream-compatible ONNX

[utils/export_yolo26.py](utils/export_yolo26.py) converts the trained model to an ONNX format compatible with the DeepStream parser. It reformats the output tensor to `[x1, y1, x2, y2, score, class]` and generates a `labels.txt` file.

Output files: `best.pt`, `best.onnx`, `labels.txt`.

### 6. Build the custom inference library

The custom YOLO parser for DeepStream lives in [nvdsinfer_custom_impl_Yolo/](nvdsinfer_custom_impl_Yolo/). Build it with:

```bash
export CUDA_VER=11.4
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

### 7. TensorRT engine generation

On first run, DeepStream automatically converts `best.onnx` into a TensorRT engine. The inference config is in [config_infer_primary_yolo26.txt](config_infer_primary_yolo26.txt) and is set up for FP16, batch size 8, with aspect-ratio-preserving padding.

### 8. Run DeepStream

```bash
deepstream-app -c deepstream_app_config.txt
```

The app config is in [deepstream_app_config.txt](deepstream_app_config.txt). Point the `uri` field at your video source before running.

---

## Benchmarks

The [benchmarks/](benchmarks/) directory contains scripts to measure model accuracy and DeepStream throughput:

| Script | What it does |
|---|---|
| [run_model_benchmarks.py](benchmarks/run_model_benchmarks.py) | Validates `.pt` and `.engine` backends; reports mAP, latency, FPS, and GPU telemetry |
| [run_deepstream_benchmarks.py](benchmarks/run_deepstream_benchmarks.py) | Runs DeepStream across stream profiles and parses FPS from logs |
| [generate_deepstream_configs.py](benchmarks/generate_deepstream_configs.py) | Auto-generates configs for 1, 2, 4, 8, and 16 concurrent streams in FP16 |
| [telemetry.py](benchmarks/telemetry.py) | Shared collector for CPU, RAM, GPU power, and GPU memory (supports `nvidia-smi` and Jetson `tegrastats`) |

---

## Prerequisites

- Python: `ultralytics`, `psutil`
- NVIDIA DeepStream runtime with `deepstream-app`
