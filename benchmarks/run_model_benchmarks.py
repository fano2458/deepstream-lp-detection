from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
np.bool = np.bool_

from ultralytics import YOLO

from telemetry import TelemetryCollector


def parse_model_spec(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid model spec '{raw}'. Use label=path")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid model spec '{raw}'. Use label=path")
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
    print(f"\n[benchmark] {label}: loading {model_path}")
    model = YOLO(model_path)

    if warmup > 0:
        print(f"[benchmark] {label}: warmup {warmup} images")
        model.predict(source="/home/jetson/Desktop/DeepStream-Yolo/video_cut.mp4", imgsz=imgsz, device=device, verbose=False, stream=False)

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

    metrics = {
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

    print(
        "[benchmark] {}: mAP50-95={:.4f}, infer_ms={:.3f}, est_fps={}".format(
            label,
            metrics["map50_95"],
            metrics["speed_inference_ms"],
            f"{est_fps:.2f}" if est_fps is not None else "n/a",
        )
    )
    return metrics


def write_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PT vs TensorRT models")
    parser.add_argument("--data", required=True, help="Dataset yaml path")
    parser.add_argument("--split", default="test", help="Dataset split: test/val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs before val")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model entry label=path. Repeat for multiple models.",
    )
    parser.add_argument("--output-json", default="benchmarks/results/model_benchmarks.json")
    parser.add_argument("--output-csv", default="benchmarks/results/model_benchmarks.csv")
    args = parser.parse_args()

    model_specs = [parse_model_spec(x) for x in args.model]

    for _, path in model_specs:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if not os.path.isfile(args.data):
        raise FileNotFoundError(args.data)

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

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(rows, args.output_csv)

    print(f"\nWrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
