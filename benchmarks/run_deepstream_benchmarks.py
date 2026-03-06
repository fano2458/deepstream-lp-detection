from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from telemetry import TelemetryCollector


PERF_LINE_RE = re.compile(r"^\*\*PERF:\s+(.+)$", re.MULTILINE)
FPS_VAL_RE = re.compile(r"(\d+\.\d+)\s+\((\d+\.\d+)\)")


def parse_fps_samples(log_text: str) -> List[float]:
    samples: List[float] = []
    for m in PERF_LINE_RE.finditer(log_text):
        payload = m.group(1)
        if "FPS" in payload:
            continue
        vals = FPS_VAL_RE.findall(payload)
        if vals:
            avg_fps = sum(float(v[0]) for v in vals) / len(vals)
            samples.append(avg_fps)
    return samples


def run_deepstream_once(config_path: str, duration_s: int, warmup_s: int) -> Dict[str, object]:
    cmd = ["deepstream-app", "-c", config_path]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    telem = TelemetryCollector(interval_s=1.0)
    telem.start()

    lines: List[str] = []
    t0 = time.time()

    try:
        while True:
            now = time.time()
            if now - t0 >= duration_s:
                break

            if proc.stdout is None:
                break

            line = proc.stdout.readline()
            if line:
                lines.append(line)
            elif proc.poll() is not None:
                break

        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        summary = telem.stop().to_dict()

    log_text = "".join(lines)
    fps_samples = parse_fps_samples(log_text)

    effective_samples = fps_samples[warmup_s:] if len(fps_samples) > warmup_s else fps_samples
    fps_avg = sum(effective_samples) / len(effective_samples) if effective_samples else None
    fps_max = max(effective_samples) if effective_samples else None

    result = {
        "config": config_path,
        "exit_code": proc.returncode,
        "fps_samples": fps_samples,
        "fps_avg": fps_avg,
        "fps_max": fps_max,
        "log_tail": "\n".join(lines[-80:]),
    }
    result.update(summary)
    return result


def write_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    if not rows:
        return
    keys = sorted([k for k in rows[0].keys() if k not in {"fps_samples", "log_tail"}])
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def discover_configs(config_dir: Path) -> List[Path]:
    return sorted(config_dir.glob("deepstream_app_fp16_s*.txt"))


def measure_idle(duration_s: int = 10) -> Dict[str, object]:
    print(f"\n[deepstream] measuring idle baseline for {duration_s}s ...")
    telem = TelemetryCollector(interval_s=1.0)
    telem.start()
    time.sleep(duration_s)
    summary = telem.stop().to_dict()
    summary["label"] = "idle"
    print(
        "[deepstream] idle: cpu_avg={:.1f}% ram_avg={:.0f}MB gpu_mem_avg={:.0f}MB".format(
            summary.get("cpu_percent_avg") or 0,
            summary.get("ram_used_mb_avg") or 0,
            summary.get("gpu_memory_mb_avg") or 0,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepStream multi-stream benchmark runner")
    parser.add_argument("--config-dir", default="benchmarks/configs")
    parser.add_argument("--duration-s", type=int, default=90)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--idle-s", type=int, default=10, help="Idle baseline measurement duration")
    parser.add_argument("--output-json", default="benchmarks/results/deepstream_benchmarks.json")
    parser.add_argument("--output-csv", default="benchmarks/results/deepstream_benchmarks.csv")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    configs = discover_configs(config_dir)
    if not configs:
        raise FileNotFoundError(f"No FP16 configs found in {config_dir}")

    idle = measure_idle(args.idle_s)

    rows: List[Dict[str, object]] = []
    for cfg in configs:
        print(f"\n[deepstream] running {cfg}")
        result = run_deepstream_once(
            config_path=str(cfg),
            duration_s=args.duration_s,
            warmup_s=args.warmup_samples,
        )

        stream_match = re.search(r"_s(\d+)\.txt$", cfg.name)
        stream_count = int(stream_match.group(1)) if stream_match else None

        result["stream_count"] = stream_count
        result["precision"] = "fp16"
        rows.append(result)

        print(
            "[deepstream] {} streams={} fps_avg={} cpu_avg={}% ram_avg={}MB gpu_mem_avg={}MB".format(
                "fp16",
                stream_count,
                f"{result['fps_avg']:.2f}" if result["fps_avg"] is not None else "n/a",
                f"{result['cpu_percent_avg']:.1f}" if result.get("cpu_percent_avg") is not None else "n/a",
                f"{result['ram_used_mb_avg']:.0f}" if result.get("ram_used_mb_avg") is not None else "n/a",
                f"{result['gpu_memory_mb_avg']:.0f}" if result.get("gpu_memory_mb_avg") is not None else "n/a",
            )
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    output = {"idle_baseline": idle, "benchmarks": rows}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_csv(rows, args.output_csv)

    print(f"\nWrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
