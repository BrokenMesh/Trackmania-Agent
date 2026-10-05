"""tools/measure_latency.py: model specs, budget arithmetic, a --quick run on the fake game."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tmagent.config import Config, load_config
from tools import measure_latency as ml

REPO = Path(__file__).resolve().parents[2]
TINY = "tiny:d_model=32,n_layers=1,n_heads=2,tokens_per_frame=4,data.history_s=0.25"


def test_parse_model_spec_forms():
    spec = ml.parse_model_spec(
        "big:d_model=512,n_layers=8,data.history_s=2.0,use_action_history=false"
    )
    assert spec.name == "big"
    assert spec.model == {"d_model": 512, "n_layers": 8, "use_action_history": False}
    assert spec.data == {"history_s": 2.0}
    # bare name = overlay of configs/<name>.yaml (model section + history_s / chunk_len)
    smoke = ml.parse_model_spec("smoke")
    assert smoke.model["d_model"] == 64 and smoke.data == {"history_s": 0.5}
    base = ml.parse_model_spec("baseline_single_frame")
    assert base.data == {"history_s": 0} and base.model == {"use_action_history": False}
    mixed = ml.parse_model_spec("m:@configs/smoke.yaml,d_model=96")
    assert mixed.model["d_model"] == 96 and mixed.model["n_layers"] == 2
    for bad in ("nosuchconfig", "x:d_model", "x:foo.bar=1"):
        with pytest.raises(ValueError):
            ml.parse_model_spec(bad)


def test_apply_spec_validates_and_does_not_touch_the_base():
    cfg = load_config(REPO / "configs" / "fake.yaml")
    out = ml.apply_spec(cfg, ml.parse_model_spec("x:d_model=128,data.chunk_len=4"))
    assert out.model.d_model == 128 and out.data.chunk_len == 4
    assert cfg.model.d_model == 64 and cfg.data.chunk_len == 8
    with pytest.raises(ValueError, match="unexpected keyword|model spec"):
        ml.apply_spec(cfg, ml.parse_model_spec("x:no_such_key=1"))
    with pytest.raises(ValueError, match="divisible"):
        ml.apply_spec(cfg, ml.parse_model_spec("x:d_model=100,n_heads=8"))


def stat(p99: float) -> dict:
    return {"n": 10, "mean_ms": p99 / 2, "p50_ms": p99 / 2, "p95_ms": p99 * 0.9, "p99_ms": p99,
            "max_ms": p99}  # fmt: skip


def row(name: str, params: int, p99: float, chunk_len: int = 8) -> dict:
    return {"name": name, "params": params, "chunk_len": chunk_len, "chain": stat(p99)}


def test_budget_arithmetic():
    cfg = Config()  # control 60 Hz, frames 20 Hz -> period 50 ms
    rows = [
        row("small", 1_000_000, 20.0),  # horizon 133.3 ms
        row("mid", 5_000_000, 60.0),  # misses the 50 ms period
        row("slowhorizon", 9_000_000, 40.0, chunk_len=4),  # horizon 66.7 ms: margin 40 % < 50 %
        row("ok_big", 3_000_000, 45.0, chunk_len=16),  # horizon 266.7 ms
    ]
    b = ml.budget(cfg, rows)
    by = {r["name"]: r for r in b["rows"]}
    s = by["small"]
    assert s["period_ms"] == 50.0 and s["horizon_ms"] == pytest.approx(8 / 60 * 1000)
    assert s["margin_ms"] == pytest.approx(8 / 60 * 1000 - 20) and s["margin_pct"] == pytest.approx(
        85
    )
    assert s["fits_period"] and s["margin_ok"] and s["fits"]
    assert s["min_chunk_len"] == math.ceil(0.020 * 60 * 2) == 3  # latency <= half the horizon
    assert s["gapless_chunk_len"] == math.ceil(0.070 * 60) == 5  # latency + one period
    assert not by["mid"]["fits_period"] and not by["mid"]["fits"]
    assert by["mid"]["min_chunk_len"] == 8  # ceil(0.06 * 60 * 2) = 7.2 -> 8
    assert by["slowhorizon"]["fits_period"] and not by["slowhorizon"]["margin_ok"]
    assert by["slowhorizon"]["min_chunk_len"] == 5 and by["ok_big"]["fits"]
    assert b["recommended"] == "ok_big"  # largest fitting model by parameters
    # never below one frame step (R = 60 / 20 = 3 actions), however fast the model is
    fast = ml.budget(cfg, [row("fast", 1, 0.5)])["rows"][0]
    assert fast["min_chunk_len"] == 3 and fast["gapless_chunk_len"] == 4  # period + 0.5 ms
    # nothing fits
    assert ml.budget(cfg, [row("huge", 1, 500.0)])["recommended"] is None
    # margin rule at the boundary: latency exactly half the horizon still fits
    edge = ml.budget(cfg, [row("edge", 1, 8 / 60 * 1000 / 2 - 1e-9)])["rows"][0]
    assert edge["margin_ok"]


def test_quick_run_on_the_fake_game(tmp_path: Path, capsys):
    out = tmp_path / "out" / "latency.md"
    argv = ["--config", str(REPO / "configs" / "fake.yaml"), "--quick", "--out", str(out)]
    argv += ["--set", "data.resolution=[32, 24]", "--models", TINY, "baseline_single_frame"]
    assert ml.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    for part in ("## Game", "grab_frame", "rt_set_action", "rt_frame_interval", "rt_frame_age"):
        assert part in text
    for part in ("## Preprocess", "## Model chain", "## Budget", "tiny", "baseline_single_frame"):
        assert part in text
    res = json.loads(out.with_suffix(".json").read_text())
    assert [m["name"] for m in res["models"]] == ["tiny", "baseline_single_frame"]
    for m in res["models"]:
        assert m["params"] > 0
        for key in ("capture", "preprocess", "observe", "predict", "chain"):
            st = m[key]
            assert (
                st["n"] == 20 and 0 < st["p50_ms"] <= st["p95_ms"] <= st["p99_ms"] <= st["max_ms"]
            )
        assert m["chain"]["p99_ms"] >= m["observe"]["p99_ms"]
    sections = res["profiler"]["sections"]
    assert sections["game/grab_frame"]["n"] == 20 and sections["game/rt_set_action"]["n"] == 200
    assert sections["game/rt_frame_interval"]["p50_ms"] == pytest.approx(16.7, abs=8)
    assert res["game"]["map"] == "fake:oval" and res["game"]["render_s_per_race_s"] > 0
    b = res["budget"]
    assert len(b["rows"]) == 2 and b["recommended"] in ("tiny", "baseline_single_frame", None)
    fitting = [r["name"] for r in b["rows"] if r["fits"]]
    assert (b["recommended"] is None) == (not fitting)  # fits depends on the machine load
    assert "written to" in capsys.readouterr().out


def test_no_game_uses_synthetic_frames(tmp_path: Path):
    out = tmp_path / "latency.md"
    argv = ["--config", str(REPO / "configs" / "fake.yaml"), "--quick", "--no-game"]
    argv += ["--out", str(out), "--models", TINY]
    assert ml.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    assert "Game part skipped" in text and "disabled (--no-game)" in text
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["models"][0]["capture"]["p99_ms"] < 1.0  # capture is a no-op without a game


def test_missing_map_for_the_tmnf_backend_skips_the_game(tmp_path: Path):
    cfg = load_config(REPO / "configs" / "tmnf.yaml", ["runtime.device=cpu"])
    spec = ml.parse_model_spec(TINY)
    res, _ = ml.run(cfg, [spec], None, n=3, warmup=1, n_grabs=3, rt_seconds=0.1)
    assert "no map given" in res["game"]["error"]
    assert res["models"][0]["chain"]["n"] == 3 and res["meta"]["backend"] == "tmnf"
