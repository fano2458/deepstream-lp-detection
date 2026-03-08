"""System telemetry collection for benchmark runs.

Collects CPU, RAM, and GPU metrics in a background thread while inference
workloads are running.  Supports two GPU backends:

- ``nvidia-smi`` — standard desktop/server GPUs
- ``tegrastats`` — NVIDIA Jetson edge devices

Example::

    collector = TelemetryCollector(interval_s=1.0)
    collector.start()
    run_workload()
    summary = collector.stop()
    print(summary.to_dict())
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil


@dataclass
class TelemetrySummary:
    """Aggregated telemetry metrics over a measurement window.

    Attributes:
        duration_s: Total elapsed time in seconds.
        cpu_percent_avg: Mean CPU utilisation across all cores (%).
        cpu_percent_max: Peak CPU utilisation (%).
        ram_used_mb_avg: Mean resident memory usage (MB).
        ram_used_mb_max: Peak resident memory usage (MB).
        gpu_memory_mb_avg: Mean GPU memory usage (MB).
        gpu_memory_mb_max: Peak GPU memory usage (MB).
    """

    duration_s: float
    cpu_percent_avg: Optional[float]
    cpu_percent_max: Optional[float]
    ram_used_mb_avg: Optional[float]
    ram_used_mb_max: Optional[float]
    gpu_memory_mb_avg: Optional[float]
    gpu_memory_mb_max: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return a flat dict representation suitable for CSV/JSON serialisation."""
        return {
            "duration_s": self.duration_s,
            "cpu_percent_avg": self.cpu_percent_avg,
            "cpu_percent_max": self.cpu_percent_max,
            "ram_used_mb_avg": self.ram_used_mb_avg,
            "ram_used_mb_max": self.ram_used_mb_max,
            "gpu_memory_mb_avg": self.gpu_memory_mb_avg,
            "gpu_memory_mb_max": self.gpu_memory_mb_max,
        }


class TelemetryCollector:
    """Background telemetry sampler.

    Spawns a daemon thread that periodically queries CPU/RAM/GPU metrics.
    Call :meth:`start` before the workload and :meth:`stop` after it to
    retrieve a :class:`TelemetrySummary`.

    Args:
        interval_s: Sampling interval in seconds (default: 1.0).
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        self._cpu: List[float] = []
        self._ram_mb: List[float] = []
        self._gpu_power_w: List[float] = []
        self._gpu_mem_mb: List[float] = []

        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._has_tegrastats = shutil.which("tegrastats") is not None

        if self._has_nvidia_smi:
            self._backend = "nvidia-smi"
        elif self._has_tegrastats:
            self._backend = "tegrastats"
        else:
            self._backend = "none"

    def start(self) -> None:
        """Begin background sampling.  No-op if already running."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        psutil.cpu_percent(interval=None)  # prime the CPU counter
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        """Stop sampling and return aggregated metrics.

        Returns:
            :class:`TelemetrySummary` with min/avg/max for each metric.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2.0))
        self._end_time = time.time()
        return self._summarize()

    # ------------------------------------------------------------------
    # Internal sampling loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            self._sample_once()
            time.sleep(self.interval_s)

    def _sample_once(self) -> None:
        self._cpu.append(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        self._ram_mb.append(vm.used / (1024.0 * 1024.0))

        gpu_power, gpu_mem = self._sample_gpu()
        if gpu_power is not None:
            self._gpu_power_w.append(gpu_power)
        if gpu_mem is not None:
            self._gpu_mem_mb.append(gpu_mem)

    def _sample_gpu(self) -> tuple[Optional[float], Optional[float]]:
        """Dispatch to the appropriate GPU backend.

        Returns:
            Tuple of ``(power_watts, memory_mb)``; values are ``None`` if
            the metric is unavailable.
        """
        if self._has_nvidia_smi:
            return self._sample_gpu_nvidia_smi()
        if self._has_tegrastats:
            return self._sample_gpu_tegrastats()
        return None, None

    def _sample_gpu_nvidia_smi(self) -> tuple[Optional[float], Optional[float]]:
        """Query GPU power draw and memory via ``nvidia-smi``.

        Returns:
            ``(power_w, memory_mb)`` or ``(None, None)`` on failure.
        """
        cmd = ["nvidia-smi", "--query-gpu=power.draw,memory.used", "--format=csv,noheader,nounits"]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0).strip()
            if not out:
                return None, None
            p_str, m_str = [x.strip() for x in out.splitlines()[0].split(",", 1)]
            return float(p_str), float(m_str)
        except Exception:
            return None, None

    def _sample_gpu_tegrastats(self) -> tuple[Optional[float], Optional[float]]:
        """Query GPU power and RAM via Jetson ``tegrastats``.

        Parses the ``POM_5V_IN`` / ``VDD_IN`` power field (converted from mW
        to W) and the ``RAM`` used field.

        Returns:
            ``(power_w, memory_mb)`` or ``(None, None)`` on failure.
        """
        cmd = ["bash", "-lc", "timeout 2 tegrastats --interval 1000 2>/dev/null | head -n 1"]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=3.0).strip()
            if not out:
                return None, None

            power_w: Optional[float] = None
            mem_mb: Optional[float] = None

            for key in ("POM_5V_IN", "VDD_IN"):
                idx = out.find(key)
                if idx >= 0:
                    digits = ""
                    for ch in out[idx: idx + 48]:
                        if ch.isdigit():
                            digits += ch
                        elif digits:
                            break
                    if digits:
                        power_w = float(digits) / 1000.0
                        break

            ram_idx = out.find("RAM ")
            if ram_idx >= 0:
                used = ""
                for ch in out[ram_idx + 4: ram_idx + 28]:
                    if ch.isdigit():
                        used += ch
                    else:
                        break
                if used:
                    mem_mb = float(used)

            return power_w, mem_mb
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        """Return the mean of *values*, or ``None`` if the list is empty."""
        return sum(values) / float(len(values)) if values else None

    @staticmethod
    def _max(values: List[float]) -> Optional[float]:
        """Return the maximum of *values*, or ``None`` if the list is empty."""
        return max(values) if values else None

    def _summarize(self) -> TelemetrySummary:
        end = self._end_time if self._end_time is not None else time.time()
        start = self._start_time if self._start_time is not None else end
        return TelemetrySummary(
            duration_s=max(0.0, end - start),
            cpu_percent_avg=self._mean(self._cpu),
            cpu_percent_max=self._max(self._cpu),
            ram_used_mb_avg=self._mean(self._ram_mb),
            ram_used_mb_max=self._max(self._ram_mb),
            gpu_memory_mb_avg=self._mean(self._gpu_mem_mb),
            gpu_memory_mb_max=self._max(self._gpu_mem_mb),
        )
