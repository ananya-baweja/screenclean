"""Environment report: hardware, Python and key package versions.

Only a fixed list of packages is reported (no full ``pip freeze``) so the
report stays short and never lists unrelated software.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

KEY_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "onnxruntime",
    "onnx",
    "pillow",
    "scikit-image",
)


def package_versions(names: tuple[str, ...] = KEY_PACKAGES) -> dict[str, str]:
    """Installed versions of ``names``; packages that aren't installed are left out."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def git_sha(repo_root: str | Path | None = None) -> str | None:
    """Current commit SHA of the repo, or None if git isn't available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() or None


def total_ram_gb() -> float | None:
    """Total RAM in GB (Linux via /proc/meminfo; None elsewhere)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        return None
    return None


def gpu_info() -> dict[str, Any]:
    """GPU details from PyTorch if it's installed; ``{"available": False}`` otherwise."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def collect_env(repo_root: str | Path | None = None, disk_path: str | Path = ".") -> dict[str, Any]:
    """Build the environment report saved with every job as ``env.json``."""
    disk = shutil.disk_usage(disk_path)
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "gpu": gpu_info(),
        "packages": package_versions(),
        "git_sha": git_sha(repo_root),
    }
