"""Unit tests for src.telemetry."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry import TelemetryCollector, TelemetrySummary


# ---------------------------------------------------------------------------
# TelemetrySummary
# ---------------------------------------------------------------------------


class TestTelemetrySummary:
    def test_to_dict_keys(self):
        summary = TelemetrySummary(
            duration_s=5.0,
            cpu_percent_avg=10.0,
            cpu_percent_max=20.0,
            ram_used_mb_avg=512.0,
            ram_used_mb_max=600.0,
            gpu_memory_mb_avg=1024.0,
            gpu_memory_mb_max=1100.0,
        )
        d = summary.to_dict()
        assert set(d.keys()) == {
            "duration_s",
            "cpu_percent_avg",
            "cpu_percent_max",
            "ram_used_mb_avg",
            "ram_used_mb_max",
            "gpu_memory_mb_avg",
            "gpu_memory_mb_max",
        }

    def test_to_dict_values(self):
        summary = TelemetrySummary(
            duration_s=3.0,
            cpu_percent_avg=50.0,
            cpu_percent_max=80.0,
            ram_used_mb_avg=None,
            ram_used_mb_max=None,
            gpu_memory_mb_avg=None,
            gpu_memory_mb_max=None,
        )
        d = summary.to_dict()
        assert d["duration_s"] == 3.0
        assert d["cpu_percent_avg"] == 50.0
        assert d["ram_used_mb_avg"] is None


# ---------------------------------------------------------------------------
# TelemetryCollector._mean / _max
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_mean_empty(self):
        assert TelemetryCollector._mean([]) is None

    def test_mean_values(self):
        assert TelemetryCollector._mean([1.0, 3.0]) == pytest.approx(2.0)

    def test_max_empty(self):
        assert TelemetryCollector._max([]) is None

    def test_max_values(self):
        assert TelemetryCollector._max([1.0, 5.0, 3.0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TelemetryCollector — backend detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_nvidia_smi_preferred(self):
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None):
            c = TelemetryCollector()
        assert c._backend == "nvidia-smi"

    def test_tegrastats_fallback(self):
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/tegrastats" if x == "tegrastats" else None):
            c = TelemetryCollector()
        assert c._backend == "tegrastats"

    def test_no_gpu_backend(self):
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector()
        assert c._backend == "none"


# ---------------------------------------------------------------------------
# TelemetryCollector — start / stop lifecycle
# ---------------------------------------------------------------------------


class TestCollectorLifecycle:
    def test_start_stop_returns_summary(self):
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector(interval_s=0.05)
        c.start()
        time.sleep(0.15)
        summary = c.stop()
        assert isinstance(summary, TelemetrySummary)
        assert summary.duration_s >= 0.0

    def test_double_start_is_noop(self):
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector(interval_s=0.05)
        c.start()
        thread_before = c._thread
        c.start()  # should be a no-op
        assert c._thread is thread_before
        c.stop()

    def test_cpu_samples_collected(self):
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector(interval_s=0.05)
        c.start()
        time.sleep(0.2)
        c.stop()
        assert len(c._cpu) >= 1

    def test_ram_samples_collected(self):
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector(interval_s=0.05)
        c.start()
        time.sleep(0.2)
        c.stop()
        assert len(c._ram_mb) >= 1
        assert all(mb > 0 for mb in c._ram_mb)


# ---------------------------------------------------------------------------
# TelemetryCollector — nvidia-smi parsing
# ---------------------------------------------------------------------------


class TestNvidiaSmiParsing:
    def _make_collector(self) -> TelemetryCollector:
        with patch("shutil.which", return_value=None):
            c = TelemetryCollector()
        c._has_nvidia_smi = True
        return c

    def test_parses_valid_output(self):
        c = self._make_collector()
        with patch("subprocess.check_output", return_value="45.23, 2048\n"):
            power, mem = c._sample_gpu_nvidia_smi()
        assert power == pytest.approx(45.23)
        assert mem == pytest.approx(2048.0)

    def test_returns_none_on_empty_output(self):
        c = self._make_collector()
        with patch("subprocess.check_output", return_value=""):
            power, mem = c._sample_gpu_nvidia_smi()
        assert power is None
        assert mem is None

    def test_returns_none_on_exception(self):
        c = self._make_collector()
        with patch("subprocess.check_output", side_effect=Exception("fail")):
            power, mem = c._sample_gpu_nvidia_smi()
        assert power is None
        assert mem is None
