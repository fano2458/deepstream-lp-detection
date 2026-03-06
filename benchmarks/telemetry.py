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
    duration_s: float
    cpu_percent_avg: Optional[float]
    cpu_percent_max: Optional[float]
    ram_used_mb_avg: Optional[float]
    ram_used_mb_max: Optional[float]
    gpu_memory_mb_avg: Optional[float]
    gpu_memory_mb_max: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
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

        self._backend = "none"
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._has_tegrastats = shutil.which("tegrastats") is not None

        if self._has_nvidia_smi:
            self._backend = "nvidia-smi"
        elif self._has_tegrastats:
            self._backend = "tegrastats"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2.0))
        self._end_time = time.time()
        return self._summarize()

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
        if self._has_nvidia_smi:
            return self._sample_gpu_nvidia_smi()
        if self._has_tegrastats:
            return self._sample_gpu_tegrastats()
        return None, None

    def _sample_gpu_nvidia_smi(self) -> tuple[Optional[float], Optional[float]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2.0).strip()
            if not out:
                return None, None
            first = out.splitlines()[0]
            p_str, m_str = [x.strip() for x in first.split(",", 1)]
            return float(p_str), float(m_str)
        except Exception:
            return None, None

    def _sample_gpu_tegrastats(self) -> tuple[Optional[float], Optional[float]]:
        cmd = [
            "bash",
            "-lc",
            "timeout 2 tegrastats --interval 1000 2>/dev/null | head -n 1",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=3.0).strip()
            if not out:
                return None, None

            power_w = None
            mem_mb = None

            for key in ("POM_5V_IN", "VDD_IN"):
                idx = out.find(key)
                if idx >= 0:
                    snippet = out[idx: idx + 48]
                    digits = ""
                    for ch in snippet:
                        if ch.isdigit():
                            digits += ch
                        elif digits:
                            break
                    if digits:
                        power_w = float(digits) / 1000.0
                        break

            ram_idx = out.find("RAM ")
            if ram_idx >= 0:
                snippet = out[ram_idx + 4: ram_idx + 28]
                used = ""
                for ch in snippet:
                    if ch.isdigit():
                        used += ch
                    else:
                        break
                if used:
                    mem_mb = float(used)

            return power_w, mem_mb
        except Exception:
            return None, None

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / float(len(values))

    @staticmethod
    def _max(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return max(values)

    def _summarize(self) -> TelemetrySummary:
        end = self._end_time if self._end_time is not None else time.time()
        start = self._start_time if self._start_time is not None else end
        duration_s = max(0.0, end - start)

        return TelemetrySummary(
            duration_s=duration_s,
            cpu_percent_avg=self._mean(self._cpu),
            cpu_percent_max=self._max(self._cpu),
            ram_used_mb_avg=self._mean(self._ram_mb),
            ram_used_mb_max=self._max(self._ram_mb),
            gpu_memory_mb_avg=self._mean(self._gpu_mem_mb),
            gpu_memory_mb_max=self._max(self._gpu_mem_mb),
        )
