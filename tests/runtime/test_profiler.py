from __future__ import annotations

import json
import threading
import time

import pytest

from tmagent.runtime.profiler import LatencyProfiler


def test_percentiles_on_known_data():
    prof = LatencyProfiler()
    for ms in range(1, 101):  # 1..100 ms
        prof.record("x", ms / 1000.0)
    st = prof.summary()["sections"]["x"]
    assert st["n"] == 100
    assert st["mean_ms"] == pytest.approx(50.5)
    assert st["p50_ms"] == pytest.approx(50.5)
    assert st["p95_ms"] == pytest.approx(95.05)
    assert st["p99_ms"] == pytest.approx(99.01)
    assert st["max_ms"] == pytest.approx(100.0)
    assert prof.stat("missing") is None


def test_bounded_memory_keeps_last_samples():
    prof = LatencyProfiler(max_samples=10)
    for ms in range(1, 101):
        prof.record("x", ms / 1000.0)
    st = prof.stat("x")
    assert st["n"] == 100  # all-time count
    assert st["max_ms"] == pytest.approx(100.0)
    assert st["p50_ms"] == pytest.approx(95.5)  # median of the last 10 samples (91..100)
    assert len(prof._buf["x"]) == 10


def test_section_times_block_and_records_on_error():
    prof = LatencyProfiler()
    with prof.section("sleep"):
        time.sleep(0.02)
    with pytest.raises(RuntimeError):
        with prof.section("boom"):
            raise RuntimeError("x")
    assert 15.0 <= prof.stat("sleep")["p50_ms"] < 500.0
    assert prof.stat("boom")["n"] == 1


def test_counters_markdown_and_dump(tmp_path):
    prof = LatencyProfiler()
    prof.record("predict", 0.01)
    prof.count("deadline_miss")
    prof.count("deadline_miss", 2)
    assert prof.counter("deadline_miss") == 3
    md = prof.to_markdown()
    assert "predict" in md and "p99 ms" in md and "deadline_miss" in md and "| 3 |" in md
    path = tmp_path / "sub" / "lat.json"
    prof.dump(path)
    data = json.loads(path.read_text())
    assert data["counters"] == {"deadline_miss": 3}
    assert data["sections"]["predict"]["n"] == 1


def test_thread_safety():
    prof = LatencyProfiler(max_samples=500)

    def work() -> None:
        for i in range(1000):
            prof.record("a", i * 1e-6)
            prof.count("c")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert prof.stat("a")["n"] == 4000
    assert prof.counter("c") == 4000
    assert len(prof._buf["a"]) == 500
