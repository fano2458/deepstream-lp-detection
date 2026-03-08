"""Auto-generate DeepStream benchmark configs for multiple stream counts.

Produces one inference config and one app config per stream count under
``configs/benchmarks/``.  Generated configs reference the repo-root model
artifacts (``best.onnx``, ``model_b*_gpu0_fp16.engine``, etc.) with absolute
paths so they can be run from any working directory.

Usage::

    python scripts/generate_deepstream_configs.py \\
        --source-uri "file:///path/to/video.mp4" \\
        --fp16-engine model_b16_gpu0_fp16.engine \\
        --streams 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

STREAM_COUNTS = [1, 2, 4, 8, 16]


def read_text(path: Path) -> str:
    """Read a text file and return its contents.

    Args:
        path: File path to read.

    Returns:
        File contents as a string.
    """
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def infer_variant(
    base_text: str,
    *,
    engine: str,
    network_mode: int,
    batch: int,
    repo_root: Path,
) -> str:
    """Produce a modified inference config for a given engine and batch size.

    Rewrites path-bearing keys (``model-engine-file``, ``onnx-file``,
    ``labelfile-path``, ``custom-lib-path``) to absolute paths rooted at
    *repo_root*, and updates ``network-mode`` and ``batch-size``.

    Args:
        base_text: Contents of the template inference config.
        engine: Engine filename relative to *repo_root*.
        network_mode: TensorRT precision mode (2 = FP16).
        batch: Batch size for this variant.
        repo_root: Absolute path to the repository root.

    Returns:
        Modified config text.
    """
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
    """Compute a near-square tiled display grid for *n* streams.

    Args:
        n: Number of concurrent video streams.

    Returns:
        Tuple of ``(rows, cols)`` for the tiled display layout.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def app_variant(
    base_text: str,
    *,
    streams: int,
    infer_config_path: str,
    source_uri: str,
    use_display: bool = False,
) -> str:
    """Produce a modified DeepStream app config for *streams* concurrent sources.

    Updates source count, URI list, streammux batch size, tiled display grid,
    and sink type (no display by default for headless benchmarking).

    Args:
        base_text: Contents of the template app config.
        streams: Number of concurrent input streams.
        infer_config_path: Path to write into the ``[primary-gie]`` section.
        source_uri: Video URI to replicate for each stream.
        use_display: If ``True``, enable tiled display output; otherwise headless.

    Returns:
        Modified config text.
    """
    rows, cols = streammux_grid(streams)
    out_lines: list[str] = []
    section = ""
    uri_emitted = False

    for raw in base_text.splitlines():
        line = raw
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped

        if section == "[source0]":
            if line.startswith("num-sources="):
                line = f"num-sources={streams}"
            elif line.startswith("uri=") or ("sample_1080p" in line and line.startswith("# uri=")):
                if not uri_emitted:
                    for _ in range(streams):
                        out_lines.append(f"uri={source_uri}")
                    uri_emitted = True
                continue
        elif section == "[streammux]":
            if line.startswith("batch-size="):
                line = f"batch-size={streams}"
        elif section == "[primary-gie]":
            if line.startswith("config-file="):
                line = f"config-file={infer_config_path}"
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
    """Generate FP16 benchmark configs for all requested stream counts.

    Reads base configs from ``configs/`` and writes variants to
    ``configs/benchmarks/``.

    Args:
        repo_root: Absolute path to the repository root.
        stream_counts: Iterable of stream counts to generate configs for.
        fp16_engine: Engine filename (relative to *repo_root*) for FP16 mode.
        source_uri: Video URI used for each stream source.
    """
    base_infer_path = repo_root / "configs" / "config_infer_primary_yolo26.txt"
    base_app_path = repo_root / "configs" / "deepstream_app_config.txt"
    out_dir = repo_root / "configs" / "benchmarks"

    infer_base = read_text(base_infer_path)
    app_base = read_text(base_app_path)

    for streams in stream_counts:
        fp16_infer_name = f"config_infer_primary_fp16_s{streams:02d}.txt"
        fp16_infer_path = out_dir / fp16_infer_name

        fp16_infer = infer_variant(
            infer_base,
            engine=fp16_engine,
            network_mode=2,
            batch=streams,
            repo_root=repo_root,
        )
        write_text(fp16_infer_path, fp16_infer)

        app_fp16 = app_variant(
            app_base,
            streams=streams,
            infer_config_path=str(fp16_infer_path),
            source_uri=source_uri,
            use_display=False,
        )
        write_text(out_dir / f"deepstream_app_fp16_s{streams:02d}.txt", app_fp16)

    print(f"Generated {len(list(stream_counts))} config pairs under {out_dir}")


def main() -> None:
    """CLI entry point for config generation."""
    parser = argparse.ArgumentParser(
        description="Generate DeepStream benchmark configs for multiple stream counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=".", type=Path, help="Repository root directory")
    parser.add_argument(
        "--streams", nargs="+", type=int, default=STREAM_COUNTS,
        help="Stream counts to generate configs for",
    )
    parser.add_argument(
        "--source-uri",
        default="file:///home/jetson/Desktop/DeepStream-Yolo/video_cut.mp4",
        help="Video URI used as input source in each config",
    )
    parser.add_argument(
        "--fp16-engine",
        default="model_b16_gpu0_fp16.engine",
        help="TensorRT FP16 engine filename (relative to repo root)",
    )
    args = parser.parse_args()

    generate(
        repo_root=args.repo_root.resolve(),
        stream_counts=args.streams,
        fp16_engine=args.fp16_engine,
        source_uri=args.source_uri,
    )


if __name__ == "__main__":
    main()
