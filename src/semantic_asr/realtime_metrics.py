"""Bounded descriptive timing metadata for realtime engineering runs.

Durations are observations, never transcript correctness scores. Full-stream
count, extrema and mean remain available after the retained-sample budget is
exhausted; quantiles then become unavailable rather than describing only an
early prefix as the complete timing distribution.
"""

from __future__ import annotations

import math


class TimingSummary:
    """Accumulate finite non-negative milliseconds with bounded sample storage."""

    def __init__(self, *, max_samples: int = 10_000) -> None:
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise TypeError("max_samples must be an integer")
        if not 1 <= max_samples <= 100_000:
            raise ValueError("max_samples must be in [1, 100000]")
        self._max_samples = max_samples
        self._samples: list[float] = []
        self._count = 0
        self._minimum: float | None = None
        self._maximum: float | None = None
        self._mean = 0.0

    def add(self, duration_ms: float) -> None:
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
            raise TypeError("duration_ms must be a number")
        try:
            duration = float(duration_ms)
        except OverflowError as exc:
            raise ValueError("duration_ms must be finite and non-negative") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_ms must be finite and non-negative")

        self._count += 1
        self._minimum = duration if self._minimum is None else min(self._minimum, duration)
        self._maximum = duration if self._maximum is None else max(self._maximum, duration)
        # A running mean avoids overflow from summing individually finite durations.
        self._mean += (duration - self._mean) / self._count
        if len(self._samples) < self._max_samples:
            self._samples.append(duration)

    def as_dict(self) -> dict[str, int | float | bool | None]:
        complete = self._count == len(self._samples)
        p50 = p95 = None
        if complete and self._samples:
            # Retain the existing benchmark's interpolated percentile convention.
            from .benchmark import _percentile

            p50 = _percentile(self._samples, 0.50)
            p95 = _percentile(self._samples, 0.95)
        return {
            "count": self._count,
            "retained": len(self._samples),
            "complete": complete,
            "minMs": self._minimum,
            "maxMs": self._maximum,
            "meanMs": self._mean if self._count else None,
            "p50Ms": p50,
            "p95Ms": p95,
        }
