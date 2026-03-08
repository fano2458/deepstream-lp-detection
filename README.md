# DeepStream-Yolo

A license plate detection pipeline that trains a YOLO26 model and deploys it with NVIDIA DeepStream for real-time, multi-stream inference on GPU hardware (tested on Jetson AGX Orin).

The pipeline covers the full workflow: dataset download, hyperparameter search, model training, TensorRT export, and DeepStream integration — with benchmarking tools to measure throughput and hardware utilisation.

**Best results:** mAP50 = 0.969 | mAP50-95 = 0.691 | 194 FPS (single stream, FP16)

---

## Repository layout

```
DeepStream-Yolo/
├── src/                          # Python library (importable)
│   ├── export.py                 # YOLO26 → DeepStream ONNX export
│   └── telemetry.py              # CPU/RAM/GPU telemetry collector
├── scripts/                      # Runnable scripts
│   ├── download_data.py          # Roboflow dataset downloader
│   ├── generate_deepstream_configs.py
│   ├── run_model_benchmarks.py
│   └── run_deepstream_benchmarks.py
├── notebooks/
│   ├── hyper_parameters_search.ipynb
│   ├── train.ipynb
│   └── validate.ipynb
├── configs/
│   ├── deepstream_app_config.txt
│   ├── config_infer_primary_yolo26.txt
│   └── benchmarks/               # Auto-generated per-stream configs
├── tests/
│   ├── test_telemetry.py
│   └── test_export_utils.py
├── results/
│   ├── training_log.csv          # Per-epoch metrics from training
│   ├── model_benchmarks.csv/json # PT vs TRT accuracy + speed
│   ├── deepstream_benchmarks.csv/json
│   └── plots/                    # Training curves, PR curve, confusion matrix
│   └── nvdsinfer_custom_impl_Yolo/   # C++ TensorRT parser plugin for DeepStream
├── models/
│   └── labels.txt
├── pyproject.toml                # Package metadata + ruff config
├── requirements.txt
└── docs/
    ├── YOLO26.md                 # Step-by-step conversion guide
    └── report.md                 # Full benchmark report
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

For development (linter + tests):

```bash
pip install -e ".[dev]"
```

### 2. Download the dataset

```bash
export ROBOFLOW_API_KEY=<your_key>   # get one free at https://app.roboflow.com
python scripts/download_data.py --dest data/
```

The dataset (`license-plate-recognition-rxg4e`, version 12, ~10 k images) is downloaded in YOLO format to `data/`. Raw data is excluded from the repository via `.gitignore`.

### 3. Build the DeepStream custom library (Jetson / Linux only)

```bash
export CUDA_VER=11.4
make -C src/nvdsinfer_custom_impl_Yolo clean && make -C src/nvdsinfer_custom_impl_Yolo
```

---

## Training

Open and run `notebooks/train.ipynb`, or use the cells below as a reference:

```python
import torch
from ultralytics import YOLO

# Fixed seed for reproducibility
torch.manual_seed(42)

model = YOLO("yolo26n.pt")
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=128,
    optimizer="AdamW",
    cos_lr=True,
    close_mosaic=10,
    flipud=0.0,
    patience=5,
    amp=True,
    workers=8,
    cache=True,
    seed=42,
    device=0,
)
```

Training results are saved to `results/` (metrics CSV, plots).

| Metric | Value |
|---|---|
| mAP50 | 0.969 |
| mAP50-95 | 0.691 |
| Precision | 0.972 |
| Recall | 0.932 |

---

## Export to ONNX (DeepStream-compatible)

```bash
python -m src.export -w best.pt --simplify --dynamic
```

This produces `best.onnx` and `labels.txt` alongside the checkpoint.
The output tensor format is `[batch, num_detections, 6]` → `[x1, y1, x2, y2, score, class_id]`.

---

## DeepStream deployment

### 1. Generate benchmark configs

```bash
python scripts/generate_deepstream_configs.py \
    --source-uri "file:///path/to/video.mp4" \
    --fp16-engine model_b16_gpu0_fp16.engine
```

### 2. Run inference

```bash
deepstream-app -c configs/deepstream_app_config.txt
```

TensorRT engines are built automatically on first run from `best.onnx`.

---

## Benchmarking

### Model accuracy + speed (PT vs TensorRT)

```bash
python scripts/run_model_benchmarks.py \
    --data data/data.yaml \
    --model "PT FP32=best.pt" \
    --model "TRT FP16=best.engine" \
    --output-csv results/model_benchmarks.csv
```

### DeepStream throughput (1 – 16 streams)

```bash
python scripts/run_deepstream_benchmarks.py \
    --config-dir configs/benchmarks \
    --output-csv results/deepstream_benchmarks.csv
```

Saved results are in [results/](results/).

| Streams | FPS (total) | FPS / stream |
|---|---|---|
| 1 | 194 | 194 |
| 4 | 290 | 72 |
| 8 | 400 | 50 |
| 16 | 483 | 30 |

---

## Tests

```bash
pytest
```

Tests cover telemetry collection, config generation, and model spec parsing.
They do not require a GPU or the Ultralytics stack.

---

## Linting

```bash
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/
```

---

## Hardware

| Component | Spec |
|---|---|
| Platform | NVIDIA Jetson AGX Orin (64 GB) |
| JetPack | 5.1.2 |
| CUDA | 11.4 |
| TensorRT | 8.5.2.2 |
| DeepStream | 6.3.0 |
