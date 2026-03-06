# DeepStream-Yolo

This section documents what was done in this repository for the YOLO26 license-plate project, based on the files currently present.

### 1. Dataset download and preparation

- Dataset script: `test_quantization/download_data.py`
- Dataset source: Roboflow `license-plate-recognition-rxg4e` (version `12`) in YOLO26 format.
- Data config used in training/validation: `/home/nailf/test_quantization/data/data.yaml`

### 2. Hyperparameter search (initial tuning)

- Notebook: `test_quantization/hyper_parameters_search.ipynb`
- Base model: `yolo26n.pt`
- Tuning method: `model.tune(...)`
- Search settings captured in notebook:
  - `epochs=5`, `iterations=10`
  - `optimizer="AdamW"`, `imgsz=640`, `batch=128`
  - Search over learning-rate, momentum, weight decay, loss weights, and augmentation ranges

### 3. Main training run

- Notebook: `test_quantization/train.ipynb`
- training args snapshot: `test_quantization/runs/detect/train/args.yaml`

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

- Results:
  - Training stopped at epoch `37` (early stop with patience)
  - Best `mAP50-95(B)=0.69126` at epoch `32`
  - Best `mAP50(B)=0.96865` at epoch `35`
  - Final epoch metrics: precision `0.97195`, recall `0.93152`, mAP50 `0.96816`, mAP50-95 `0.69035`

### 4. Validation and TensorRT export checks

  - `test_quantization/validate.ipynb`
- Validation flow used:
  - Validate `best.pt`
  - Export TensorRT engine path in notebook (`model.export(format="engine", dynamic=True, batch=8, workspace=4, half=True)`)
  - Re-load and validate exported engine (`best.engine`)

### 5. Export to DeepStream-compatible ONNX
  - `utils/export_yolo26.py`
- Purpose of exporter:
  - Applies YOLO26/Ultralytics export adjustments for DeepStream parser compatibility
  - Produces `labels.txt`
  - Produces ONNX output with DeepStream-friendly output tensor format `[x1, y1, x2, y2, score, class]`
- Produced artifacts:
  - `best.pt`
  - `best.onnx`
  - `labels.txt`


### 6. Build DeepStream custom YOLO inference library
- Source directory: `nvdsinfer_custom_impl_Yolo/`
- Build command:

```bash
export CUDA_VER=11.4
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```


### 7. TensorRT engine generation for deployment profiles
- Engine generation is done by DeepStream on first run using `onnx-file=best.onnx`.
- Engine artifacts currently present:
  - `model_b1_gpu0_fp32.engine`
  - `model_b8_gpu0_fp32.engine`
  - `model_b8_gpu0_fp16.engine`
- Config linkage is in `config_infer_primary_yolo26.txt`:
  - `onnx-file=best.onnx`
  - `model-engine-file=model_b8_gpu0_fp16.engine`
  - `batch-size=8`
  - `network-mode=2` (FP16)
  - `cluster-mode=4` (no NMS required for YOLO26 export)
  - `maintain-aspect-ratio=1`, `symmetric-padding=1`

### 8. DeepStream app integration and runtime

- App config: `deepstream_app_config.txt`
- Primary GIE config reference:
  - `[primary-gie] config-file=config_infer_primary_yolo26.txt`
- Input source configured:
  - `file:///home/jetson/Desktop/DeepStream-Yolo/video_cut.mp4`
- Run command:

```bash
deepstream-app -c deepstream_app_config.txt
```

### Benchmarks

- `benchmarks/run_model_benchmarks.py`
  - Runs `ultralytics` validation for multiple model backends (`.pt`, `.engine`) on the same split.
  - Captures mAP/precision/recall, latency (pre/infer/post), estimated FPS.
  - Captures telemetry: CPU, RAM, GPU power, GPU memory.

- `benchmarks/run_deepstream_benchmarks.py`
  - Runs `deepstream-app` configs for each stream profile.
  - Parses FPS samples from DeepStream logs.
  - Captures telemetry: CPU, RAM, GPU power, GPU memory.

- `benchmarks/generate_deepstream_configs.py`
  - Auto-generates DeepStream app+infer configs for stream counts `1,2,4,8,16`.
  - Generates FP16 variants. 

- `benchmarks/telemetry.py`
  - Shared telemetry collector.
  - Uses `nvidia-smi` when available, otherwise Jetson `tegrastats`.

### Prerequisites

- Python packages: `ultralytics`, `psutil`
- DeepStream runtime with `deepstream-app`
