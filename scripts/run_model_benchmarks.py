"""Benchmark PyTorch (.pt) and TensorRT (.engine) model variants.

Runs Ultralytics validation on a dataset split for each model and captures
accuracy metrics (mAP), latency, estimated FPS, and system telemetry.

Usage::

    python scripts/run_model_benchmarks.py \\
        --data data/data.yaml \\
        --model "PT FP32=best.pt" \\
        --model "TRT FP16=best.engine" \\
        --output-json results/model_benchmarks.json \\
        --output-csv  results/model_benchmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

np.bool = np.bool_  # ultralytics compatibility shim

from ultralytics import YOLO

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry import TelemetryCollector


def parse_model_spec(raw: str) -> Tuple[str, str]:
    """Parse a ``label=path`` model specification string.

    Args:
        raw: String in the format ``"Label=path/to/model.pt"``.

    Returns:
        Tuple of ``(label, path)``.

    Raises:
        ValueError: If the format is invalid or either field is empty.
    """
    if "=" not in raw:
        raise ValueError(f"Invalid model spec '{raw}'. Expected format: label=path")
    label, path = raw.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise ValueError(f"Invalid model spec '{raw}'. Both label and path must be non-empty")
    return label, path


def run_single_model(
    label: str,
    model_path: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    warmup: int,
    device: str,
) -> Dict[str, object]:
    """Validate one model and collect accuracy + telemetry metrics.

    Args:
        label: Human-readable name for this model variant (used in output).
        model_path: Path to the model file (``.pt`` or ``.engine``).
        data_yaml: Path to the Ultralytics dataset YAML file.
        split: Dataset split to validate on (``"test"`` or ``"val"``).
        imgsz: Inference image size in pixels (square).
        batch: Validation batch size.
        warmup: Number of warmup predictions before the timed run.
        device: Torch/CUDA device identifier (e.g. ``"0"`` or ``"cpu"``).

    Returns:
        Dict with accuracy metrics, speed metrics, and telemetry summary.
    """
    print(f"\n[benchmark] {label}: loading {model_path}")
    model = YOLO(model_path)

    if warmup > 0:
        print(f"[benchmark] {label}: warmup ({warmup} pass)")
        model.predict(source=model_path, imgsz=imgsz, device=device, verbose=False, stream=False)

    telem = TelemetryCollector(interval_s=1.0)
    telem.start()
    t0 = time.time()
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
        save_json=False,
    )
    elapsed = time.time() - t0
    summary = telem.stop().to_dict()

    speed = results.speed or {}
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    total_ms = preprocess + inference + postprocess
    est_fps = (1000.0 / total_ms) if total_ms > 0 else None

    metrics: Dict[str, object] = {
        "label": label,
        "path": model_path,
        "elapsed_s": elapsed,
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "map75": float(results.box.map75),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "speed_preprocess_ms": preprocess,
        "speed_inference_ms": inference,
        "speed_postprocess_ms": postprocess,
        "speed_total_ms": total_ms,
        "estimated_fps": est_fps,
    }
    metrics.update(summary)

    fps_str = f"{est_fps:.2f}" if est_fps is not None else "n/a"
    print(
        f"[benchmark] {label}: mAP50-95={metrics['map50_95']:.4f}  "
        f"infer_ms={metrics['speed_inference_ms']:.3f}  est_fps={fps_str}"
    )
    return metrics


def write_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    """Write benchmark results to a CSV file.

    Args:
        rows: List of result dicts (all must share the same keys).
        output_csv: Destination file path.
    """
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point for model benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark PT vs TensorRT models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--split", default="test", help="Dataset split: test or val")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs before validation")
    parser.add_argument(
        "--model", action="append", required=True,
        help="Model entry as 'Label=path'. Repeat for multiple models.",
    )
    parser.add_argument("--output-json", default="results/model_benchmarks.json")
    parser.add_argument("--output-csv", default="results/model_benchmarks.csv")
    args = parser.parse_args()

    model_specs = [parse_model_spec(x) for x in args.model]

    for _, path in model_specs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    rows: List[Dict[str, object]] = []
    for label, path in model_specs:
        rows.append(
            run_single_model(
                label=label,
                model_path=path,
                data_yaml=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                warmup=args.warmup,
                device=args.device,
            )
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(rows, args.output_csv)

    print(f"\nResults written to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
