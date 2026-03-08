"""Unit tests for pure utility functions in scripts/ and src/export.py.

These tests exercise functions that do not require a GPU, ONNX model, or
the Ultralytics stack — so they run anywhere (CI, developer laptop, etc.).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_deepstream_configs import (
    STREAM_COUNTS,
    app_variant,
    infer_variant,
    streammux_grid,
)
from scripts.run_model_benchmarks import parse_model_spec


# ---------------------------------------------------------------------------
# parse_model_spec
# ---------------------------------------------------------------------------


class TestParseModelSpec:
    def test_valid_spec(self):
        label, path = parse_model_spec("PT FP32=best.pt")
        assert label == "PT FP32"
        assert path == "best.pt"

    def test_path_with_equals_sign(self):
        # path itself contains '=' — only the first '=' is used as separator
        label, path = parse_model_spec("model=path/to/file=1.pt")
        assert label == "model"
        assert path == "path/to/file=1.pt"

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_model_spec("noequalssign")

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_model_spec("=path.pt")

    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_model_spec("label=")


# ---------------------------------------------------------------------------
# streammux_grid
# ---------------------------------------------------------------------------


class TestStreammuxGrid:
    @pytest.mark.parametrize(
        "n, expected_product_gte_n",
        [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)],
    )
    def test_grid_covers_streams(self, n, expected_product_gte_n):
        rows, cols = streammux_grid(n)
        assert rows * cols >= expected_product_gte_n

    def test_grid_is_near_square(self):
        for n in STREAM_COUNTS:
            rows, cols = streammux_grid(n)
            # cols should be ceil(sqrt(n))
            assert cols == math.ceil(math.sqrt(n))

    def test_single_stream(self):
        assert streammux_grid(1) == (1, 1)

    def test_four_streams(self):
        assert streammux_grid(4) == (2, 2)

    def test_sixteen_streams(self):
        assert streammux_grid(16) == (4, 4)


# ---------------------------------------------------------------------------
# infer_variant
# ---------------------------------------------------------------------------

SAMPLE_INFER_CONFIG = """\
[property]
gpu-id=0
onnx-file=best.onnx
model-engine-file=model_b8_gpu0_fp16.engine
labelfile-path=models/labels.txt
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
batch-size=8
network-mode=2
#int8-calib-file=calib.table
"""


class TestInferVariant:
    def _run(self, **kwargs) -> str:
        defaults = dict(
            engine="model_b16_gpu0_fp16.engine",
            network_mode=2,
            batch=16,
            repo_root=Path("/repo"),
        )
        defaults.update(kwargs)
        return infer_variant(SAMPLE_INFER_CONFIG, **defaults)

    def test_batch_size_updated(self):
        out = self._run(batch=4)
        assert "batch-size=4" in out

    def test_network_mode_updated(self):
        out = self._run(network_mode=2)
        assert "network-mode=2" in out

    def test_engine_path_absolute(self):
        out = self._run(engine="model_b16_gpu0_fp16.engine", repo_root=Path("/repo"))
        assert "model-engine-file=/repo/model_b16_gpu0_fp16.engine" in out

    def test_onnx_path_absolute(self):
        out = self._run(repo_root=Path("/repo"))
        assert "onnx-file=/repo/best.onnx" in out

    def test_calib_file_commented_out(self):
        out = self._run()
        assert "#int8-calib-file=calib.table" in out
        assert out.count("int8-calib-file=") == 1  # only the commented version


# ---------------------------------------------------------------------------
# app_variant
# ---------------------------------------------------------------------------

SAMPLE_APP_CONFIG = """\
[application]
enable-perf-measurement=1

[tiled-display]
enable=1
rows=4
columns=4

[source0]
enable=1
type=3
num-sources=16
uri=file:///old/video.mp4

[streammux]
batch-size=16

[primary-gie]
enable=1
config-file=config_infer_primary_yolo26.txt

[sink0]
enable=1
type=2
sync=0
"""


class TestAppVariant:
    def _run(self, streams: int = 4, **kwargs) -> str:
        defaults = dict(
            infer_config_path="configs/benchmarks/config_infer_primary_fp16_s04.txt",
            source_uri="file:///video.mp4",
            use_display=False,
        )
        defaults.update(kwargs)
        return app_variant(SAMPLE_APP_CONFIG, streams=streams, **defaults)

    def test_num_sources_updated(self):
        out = self._run(streams=2)
        assert "num-sources=2" in out

    def test_correct_number_of_uris(self):
        out = self._run(streams=4)
        uri_count = out.count("uri=file:///video.mp4")
        assert uri_count == 4

    def test_streammux_batch_updated(self):
        out = self._run(streams=8)
        assert "batch-size=8" in out

    def test_infer_config_path_updated(self):
        out = self._run(infer_config_path="configs/benchmarks/config_infer_primary_fp16_s04.txt")
        assert "config-file=configs/benchmarks/config_infer_primary_fp16_s04.txt" in out

    def test_no_display_sink_type(self):
        out = self._run(use_display=False)
        assert "type=1" in out  # FakeSink / no display

    def test_display_sink_type(self):
        out = self._run(use_display=True)
        assert "type=2" in out  # EglSink

    def test_tiled_display_grid_correct(self):
        out = self._run(streams=4)
        assert "rows=2" in out
        assert "columns=2" in out
