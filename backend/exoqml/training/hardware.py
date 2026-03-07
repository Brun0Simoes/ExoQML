from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None


@dataclass(slots=True)
class HardwareProfile:
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_total_gb: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_total_gb: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def detect_hardware(path_for_disk: Path) -> HardwareProfile:
    usage = shutil.disk_usage(path_for_disk)
    gpu_available = False
    gpu_name: str | None = None
    gpu_vram_total_gb: float | None = None

    if HAS_TORCH and torch.cuda.is_available():
        gpu_available = True
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gpu_name = props.name
        gpu_vram_total_gb = props.total_memory / (1024**3)

    vm = psutil.virtual_memory()
    return HardwareProfile(
        cpu_cores_physical=psutil.cpu_count(logical=False) or max(1, os.cpu_count() or 1),
        cpu_cores_logical=psutil.cpu_count(logical=True) or max(1, os.cpu_count() or 1),
        ram_total_gb=vm.total / (1024**3),
        ram_available_gb=vm.available / (1024**3),
        disk_total_gb=usage.total / (1024**3),
        disk_used_gb=usage.used / (1024**3),
        disk_free_gb=usage.free / (1024**3),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
    )


def recommended_num_workers(logical_cores: int) -> int:
    if logical_cores <= 4:
        return 2
    return min(24, max(4, logical_cores - 2))
