"""
Wave-state telemetry — bounded streaming statistics for market tick ingestion.

Mirrors patterns from wavetheory: orthogonal memory via fixed ring buffers.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Deque, Iterable, Optional

import numpy as np


@dataclasses.dataclass
class TickSample:
    timestamp_ns: int
    price: float
    volume: float


class WaveTelemetryEngine:
    """Fixed-capacity ring buffer with streaming mean/variance (Welford)."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 8:
            raise ValueError("capacity must be >= 8")
        self.capacity = capacity
        self._buffer: Deque[TickSample] = deque(maxlen=capacity)
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def ingest(self, sample: TickSample) -> None:
        self._buffer.append(sample)
        self._count += 1
        delta = sample.price - self._mean
        self._mean += delta / self._count
        delta2 = sample.price - self._mean
        self._m2 += delta * delta2

    def ingest_batch(self, samples: Iterable[TickSample]) -> None:
        for s in samples:
            self.ingest(s)

    @property
    def sample_count(self) -> int:
        return self._count

    def price_variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    def price_std(self) -> float:
        return float(np.sqrt(self.price_variance()))

    def latest_price(self) -> Optional[float]:
        if not self._buffer:
            return None
        return self._buffer[-1].price

    def z_score(self, price: float) -> float:
        std = self.price_std()
        if std < 1e-12:
            return 0.0
        return (price - self._mean) / std
