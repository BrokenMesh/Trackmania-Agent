"""Per-component latency profiler: named timing sections, event counters, p50/p99.

Thread-safe. Memory is bounded: every name keeps only its last `max_samples`
samples (ring buffer); `n`, `mean_ms` and `max_ms` cover all samples ever recorded.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

_FIELDS = ("n", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")


class _Section:
    """Context manager that records the elapsed perf_counter time of its block."""

    __slots__ = ("_name", "_prof", "_t0")

    def __init__(self, prof: LatencyProfiler, name: str) -> None:
        self._prof = prof
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> _Section:
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._prof.record(self._name, time.perf_counter() - self._t0)


class LatencyProfiler:
    """Collects timings (seconds in, milliseconds out) and event counters."""

    def __init__(self, max_samples: int = 100_000) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self._max = max_samples
        self._lock = threading.Lock()
        self._buf: dict[str, deque[float]] = {}
        self._tot: dict[str, list[float]] = {}  # name -> [n, sum_s, max_s]
        self._counters: dict[str, int] = {}

    def section(self, name: str) -> _Section:
        """Context manager timing its block with perf_counter."""
        return _Section(self, name)

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            buf = self._buf.get(name)
            if buf is None:
                buf = self._buf[name] = deque(maxlen=self._max)
                self._tot[name] = [0.0, 0.0, float("-inf")]
            buf.append(seconds)
            tot = self._tot[name]
            tot[0] += 1
            tot[1] += seconds
            if seconds > tot[2]:
                tot[2] = seconds

    def count(self, name: str, n: int = 1) -> None:
        """Increment an event counter (e.g. deadline_miss)."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def stat(self, name: str) -> dict[str, float] | None:
        """n, mean_ms, p50_ms, p95_ms, p99_ms, max_ms of one section, None if unknown."""
        with self._lock:
            buf = self._buf.get(name)
            if buf is None:
                return None
            data = np.fromiter(buf, dtype=np.float64, count=len(buf))
            n, total, peak = self._tot[name]
        p50, p95, p99 = np.percentile(data, [50, 95, 99]) * 1e3
        return {
            "n": int(n),
            "mean_ms": total / n * 1e3,
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "max_ms": peak * 1e3,
        }

    def summary(self) -> dict[str, Any]:
        """{"sections": {name: stat}, "counters": {name: int}}."""
        with self._lock:
            names = sorted(self._buf)
            counters = dict(sorted(self._counters.items()))
        sections = {}
        for name in names:
            st = self.stat(name)
            if st is not None:
                sections[name] = st
        return {"sections": sections, "counters": counters}

    def to_markdown(self) -> str:
        s = self.summary()
        head = ["section", "n", "mean ms", "p50 ms", "p95 ms", "p99 ms", "max ms"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for name, st in s["sections"].items():
            cells = [name, str(st["n"])] + [f"{st[k]:.3f}" for k in _FIELDS[1:]]
            lines.append("| " + " | ".join(cells) + " |")
        if s["counters"]:
            lines += ["", "| counter | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in s["counters"].items()]
        return "\n".join(lines) + "\n"

    def dump(self, path: str | Path) -> None:
        """Write the summary as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), indent=2))
