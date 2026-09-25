"""Timing helpers: a simple stopwatch and a wall-clock deadline for job budgets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class Timer:
    """Context manager that measures elapsed wall-clock seconds.

    >>> with Timer() as t:
    ...     pass
    >>> t.seconds >= 0
    True
    """

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start


@dataclass
class Deadline:
    """A wall-clock budget. Long tasks poll ``expired()`` and checkpoint before it runs out."""

    budget_s: float
    start: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return self.budget_s - self.elapsed()

    def expired(self, margin_s: float = 0.0) -> bool:
        """True when less than ``margin_s`` seconds of the budget are left."""
        return self.remaining() <= margin_s
