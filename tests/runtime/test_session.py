from __future__ import annotations

import sys
import time

import numpy as np

from tmagent.config import Config
from tmagent.interfaces import NEUTRAL_ACTION
from tmagent.runtime.session import LiveSession

from .test_helpers import DummyPolicy, FakeRealtimeGame


def test_slow_policy_never_stalls_the_control_loop():
    cfg = Config()
    game = FakeRealtimeGame(fps=60.0).start_frames()
    policy = DummyPolicy(predict_s=0.1)  # 100 ms per predict, 6 control periods
    switch = sys.getswitchinterval()
    try:
        with LiveSession(game, policy, cfg) as sess:
            time.sleep(3.0)
            st = sess.stats()
        assert sys.getswitchinterval() == switch  # restored by stop()
    finally:
        game.close()
    print(
        f"\nsession: slots={st['slots']} ticks={st['ticks']} miss_pct={st['miss_pct']:.2f} "
        f"jitter p50/p99={st['jitter_p50_ms']:.3f}/{st['jitter_p99_ms']:.3f} ms "
        f"predict p50={st['profiler']['sections']['predict']['p50_ms']:.1f} ms "
        f"observed={st['inference']['frames_observed']}"
    )
    # the loop kept its 60 Hz cadence while predict() blocked its own thread
    assert st["slots"] >= 150
    assert st["miss_pct"] < 5.0  # ~0 in practice; 50+ % would mean the loop waits for the model
    t = np.array([x for x, _ in game.actions])
    assert np.diff(t).max() < 0.08  # never a gap close to the 100 ms predict time
    # inference ran at the model's pace and its sections were profiled
    sec = st["profiler"]["sections"]
    assert 90.0 <= sec["predict"]["p50_ms"] < 400.0
    assert 5 <= st["inference"]["frames_observed"] <= 31
    assert {"observe", "predict", "frame_age", "control_jitter", "control_set_action"} <= set(sec)
    assert st["profiler"]["counters"]["control_tick"] == st["ticks"]
    # actions followed the chunks (steer derived from the image values) ...
    steers = [a.steer for _, a in game.actions]
    assert max(steers) > 0.0 and any(a.gas == 1.0 for _, a in game.actions)
    # ... and stop() ended with a neutral action
    assert game.actions[-1][1] == NEUTRAL_ACTION
    assert policy.resets == 1


def test_context_manager_and_double_stop():
    cfg = Config()
    game = FakeRealtimeGame(fps=60.0).start_frames()
    sess = LiveSession(game, DummyPolicy(), cfg)
    try:
        sess.stop()  # stop before start is a no-op
        sess.start()
        sess.start()  # idempotent
        time.sleep(0.3)
        sess.stop()
        n = len(game.actions)
        sess.stop()
        assert len(game.actions) == n
        assert game.actions[-1][1] == NEUTRAL_ACTION
        assert sess.stats()["slots"] > 0
    finally:
        game.close()
