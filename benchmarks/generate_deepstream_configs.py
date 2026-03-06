from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

STREAM_COUNTS = [1, 2, 4, 8, 16]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def infer_variant(base_text: str, *, engine: str, network_mode: int, batch: int, repo_root: Path) -> str:
    lines = []
    for raw in base_text.splitlines():
        line = raw
        if line.startswith("model-engine-file="):
            line = f"model-engine-file={repo_root / engine}"
        elif line.startswith("onnx-file="):
            val = line.split("=", 1)[1]
            line = f"onnx-file={repo_root / val}"
        elif line.startswith("labelfile-path="):
            val = line.split("=", 1)[1]
            line = f"labelfile-path={repo_root / val}"
        elif line.startswith("custom-lib-path="):
            val = line.split("=", 1)[1]
            line = f"custom-lib-path={repo_root / val}"
        elif line.startswith("network-mode="):
            line = f"network-mode={network_mode}"
        elif line.startswith("batch-size="):
            line = f"batch-size={batch}"
        elif line.startswith("int8-calib-file=") or line.startswith("#int8-calib-file="):
            line = "#int8-calib-file=calib.table"
        lines.append(line)

    return "\n".join(lines) + "\n"


def streammux_grid(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def app_variant(base_text: str, *, streams: int, infer_config_relpath: str, source_uri: str, use_display: bool) -> str:
    rows, cols = streammux_grid(streams)
    out_lines = []
    section = ""
    uri_emitted = False

    for raw in base_text.splitlines():
        line = raw
        striped = raw.strip()
        if striped.startswith("[") and striped.endswith("]"):
            section = striped

        if section == "[source0]":
            if line.startswith("num-sources="):
                line = f"num-sources={streams}"
            elif line.startswith("uri=") or (line.startswith("# uri=") and "sample_1080p" in line):
                if not uri_emitted:
                    # Emit exactly `streams` uri lines
                    for _ in range(streams):
                        out_lines.append(f"uri={source_uri}")
                    uri_emitted = True
                continue
        elif section == "[streammux]":
            if line.startswith("batch-size="):
                line = f"batch-size={streams}"
        elif section == "[primary-gie]":
            if line.startswith("config-file="):
                line = f"config-file={infer_config_relpath}"
        elif section == "[tiled-display]":
            if line.startswith("rows="):
                line = f"rows={rows}"
            elif line.startswith("columns="):
                line = f"columns={cols}"
            elif line.startswith("enable="):
                line = "enable=1" if use_display else "enable=0"
        elif section == "[sink0]":
            if line.startswith("type="):
                line = "type=2" if use_display else "type=1"
            elif line.startswith("sync="):
                line = "sync=0"
        out_lines.append(line)

    return "\n".join(out_lines) + "\n"


def generate(
    repo_root: Path,
    stream_counts: Iterable[int],
    fp16_engine: str,
    source_uri: str,
) -> None:
    base_infer_path = repo_root / "config_infer_primary_yolo26.txt"
    base_app_path = repo_root / "deepstream_app_config.txt"
    out_dir = repo_root / "benchmarks" / "configs"

    infer_base = read_text(base_infer_path)
    app_base = read_text(base_app_path)

    for streams in stream_counts:
        fp16_infer = infer_variant(
            infer_base,
            engine=fp16_engine,
            network_mode=2,
            batch=streams,
            repo_root=repo_root,
        )
        fp16_infer_name = f"config_infer_primary_fp16_s{streams:02d}.txt"
        write_text(out_dir / fp16_infer_name, fp16_infer)

        app_fp16 = app_variant(
            app_base,
            streams=streams,
            infer_config_relpath=fp16_infer_name,
            source_uri=source_uri,
            use_display=False,
        )
        write_text(out_dir / f"deepstream_app_fp16_s{streams:02d}.txt", app_fp16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepStream benchmark configs")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--streams", nargs="+", type=int, default=STREAM_COUNTS)
    parser.add_argument("--source-uri", default="file:///home/jetson/Desktop/DeepStream-Yolo/video_cut.mp4")
    parser.add_argument("--fp16-engine", default="model_b16_gpu0_fp16.engine")
    args = parser.parse_args()

    generate(
        repo_root=Path(args.repo_root).resolve(),
        stream_counts=args.streams,
        fp16_engine=args.fp16_engine,
        source_uri=args.source_uri,
    )

    print("Generated configs under benchmarks/configs")


if __name__ == "__main__":
    main()
