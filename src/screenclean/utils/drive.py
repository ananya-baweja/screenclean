"""Google Drive folder layout and crash-safe file writes.

On Colab, Drive is mounted at ``/content/drive`` and the project root is
``/content/drive/MyDrive/screenclean``. Locally (tests) any folder works.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text to a temp file, then rename it, so readers never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as pretty JSON with :func:`atomic_write_text`."""
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    """Read a JSON file, or return ``default`` if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DriveLayout:
    """Standard folders under the Drive project root."""

    root: Path

    @property
    def job_status(self) -> Path:
        return self.root / "job_status"

    @property
    def results_zips(self) -> Path:
        return self.root / "results_zips"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    def ensure(self) -> DriveLayout:
        """Create the standard folders if they are missing."""
        for p in (self.job_status, self.results_zips, self.runs, self.data):
            p.mkdir(parents=True, exist_ok=True)
        return self
