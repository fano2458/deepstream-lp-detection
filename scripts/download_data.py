"""Download the license plate dataset from Roboflow.

The dataset used in this project is:
    Roboflow Universe — license-plate-recognition-rxg4e, version 12
    https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

Usage::

    pip install roboflow
    python scripts/download_data.py --api-key <YOUR_ROBOFLOW_API_KEY> --dest data/

The downloaded dataset will be placed under ``--dest`` in YOLO format,
ready for use with the training notebook or ``scripts/run_model_benchmarks.py``.

Note:
    Large raw dataset files are excluded from the repository via ``.gitignore``.
    This script is the single source-of-truth for obtaining the data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def download(api_key: str, dest: Path, version: int = 12) -> None:
    """Download the Roboflow license plate dataset.

    Args:
        api_key: Your Roboflow API key (get one at https://app.roboflow.com).
        dest: Directory where the dataset will be saved.
        version: Dataset version to download (default: 12).
    """
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("roboflow package not found. Install it with: pip install roboflow") from exc

    dest.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    dataset_version = project.version(version)
    dataset_version.download("yolov8", location=str(dest))
    print(f"\nDataset downloaded to: {dest}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace with ``api_key``, ``dest``, and ``version``.
    """
    parser = argparse.ArgumentParser(
        description="Download license plate dataset from Roboflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--dest",
        default="data/",
        type=Path,
        help="Destination directory for the downloaded dataset",
    )
    parser.add_argument(
        "--version",
        default=12,
        type=int,
        help="Roboflow dataset version to download",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset downloader."""
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Roboflow API key is required.\n"
            "  Pass it with --api-key or set the ROBOFLOW_API_KEY environment variable.\n"
            "  Get a free key at https://app.roboflow.com"
        )
    download(api_key=args.api_key, dest=args.dest, version=args.version)


if __name__ == "__main__":
    main()
