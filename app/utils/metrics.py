from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class MetricsCollector:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._durations: Dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._durations[name] = self._durations.get(name, 0.0) + elapsed

    def add_duration(self, name: str, duration_ms: float) -> None:
        self._durations[name] = self._durations.get(name, 0.0) + duration_ms

    def merge(self, other: Dict[str, int]) -> None:
        for key, value in other.items():
            self._durations[key] = self._durations.get(key, 0.0) + float(value)

    def as_ints(self) -> Dict[str, int]:
        return {key: int(round(value)) for key, value in self._durations.items()}

    def total_elapsed_ms(self) -> int:
        return int(round((time.perf_counter() - self._start) * 1000.0))
