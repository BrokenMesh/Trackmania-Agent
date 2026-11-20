"""Reward-weighted RL data: sampling, rewards, advantage weights, weighted storage and loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tmagent.config import load_config
from tmagent.data.dataset import build_window, check_alignment, source_weight
from tmagent.data.episode_io import load_episode, save_episode
from tmagent.data.render import render_episode
from tmagent.eval.progress import ReferencePath
from tmagent.eval.rl import (
    Attempt,
    RLConfig,
    advantage_weights,
    collect_attempt,
    discounted_returns,
    sample_rows,
    step_rewards,
)
from tmagent.game import make_game
from tmagent.game.fake import scripted_driver
from tmagent.model.losses import bc_loss
from tmagent.model.policy import steer_bin_centers

MAP = "fake:s_curve"


@pytest.fixture
def setup():
    cfg = load_config("configs/fake.yaml", ["data.resolution=[32, 24]", "data.history_s=0.25"])
    game = make_game(cfg.game, cfg.data)
    tl, _ = scripted_driver(MAP, cfg.data)
    ep = render_episode(game, tl, MAP, cfg.data, {})
    return cfg, game, ep


class GasPolicy:
    """Raw outputs that say: no steer, full gas, no brake (with certainty)."""

    def __init__(self, bins: int, chunk: int = 8) -> None:
        centers = steer_bin_centers(bins).numpy()
        self.logits = np.where(np.abs(centers) == np.abs(centers).min(), 20.0, -20.0)
        self.chunk = chunk

    def reset(self) -> None:
        self.observed = 0

    def observe(self, image, past_actions) -> None:
        self.observed += 1

    def predict_outputs(self) -> dict[str, np.ndarray]:
        c = self.chunk
        return {
            "steer_logits": np.tile(self.logits, (c, 1)).astype(np.float32),
            "gas": np.full(c, 20.0, np.float32),
            "brake": np.full(c, -20.0, np.float32),
        }


def test_sample_rows_follows_the_probabilities():
    rng = np.random.default_rng(0)
    certain = {
        "steer_logits": np.array([[-20.0, -20.0, 20.0]] * 4, np.float32),
        "gas": np.full(4, 20.0, np.float32),
        "brake": np.full(4, -20.0, np.float32),
    }
    rows = sample_rows(certain, 3, rng, 1.0)
    assert rows.tolist() == [[1.0, 1.0, 0.0]] * 3

    even = {"steer_logits": np.zeros((3, 3), np.float32), "gas": np.zeros(3), "brake": np.zeros(3)}
    draws = np.concatenate([sample_rows(even, 3, rng, 1.0) for _ in range(400)])
    assert set(np.unique(draws[:, 0])) == {-1.0, 0.0, 1.0}
    assert 0.4 < draws[:, 1].mean() < 0.6


def test_collect_attempt_labels_are_the_executed_actions(setup):
    cfg, game, ep = setup
    rcfg = RLConfig()
    policy = GasPolicy(cfg.model.steer_bins)
    a = collect_attempt(game, policy, MAP, cfg, 0, ReferencePath(ep.positions), rcfg)
    check_alignment(a.episode, cfg.data)
    assert policy.observed == len(a.episode.frames)
    assert a.episode.actions[:, 1].min() == 1.0 and a.episode.actions[:, 0].max() == 0.0
    assert len(a.progress_m) == len(a.episode.actions) and a.progress_m[-1] > a.progress_m[0]
    assert a.reason in ("finished", "stuck", "offtrack", "stall", "timeout")


def _attempt(progress_m: np.ndarray, reason: str) -> Attempt:
    return Attempt(episode=None, progress_m=progress_m, reason=reason, progress=0.0, seconds=0.0)  # type: ignore[arg-type]


def test_rewards_and_returns():
    rcfg = RLConfig(crash_penalty_m=50.0, max_step_m=5.0)
    r = step_rewards(np.array([0.0, 1.0, 101.0, 102.0]), "stuck", rcfg)
    assert r.tolist() == [1.0, 5.0, 1.0, -50.0]  # the 100 m projection jump is clipped
    g = discounted_returns(np.array([1.0, 1.0]), control_hz=1, horizon_s=2.0)
    assert g.tolist() == [1.5, 1.0]


def test_faster_attempts_get_higher_weights_and_crashes_lower():
    rcfg = RLConfig()
    fast = _attempt(np.arange(0, 300, 2.0), "timeout")
    slow = _attempt(np.arange(0, 150, 1.0), "timeout")
    crash = _attempt(np.arange(0, 150, 1.0), "offtrack")
    w_fast, w_slow, w_crash = advantage_weights([fast, slow, crash], 60, rcfg)
    assert w_fast[:50].mean() > w_slow[:50].mean()
    assert w_crash[-30:].mean() < w_slow[-30:].mean()
    assert np.concatenate([w_fast, w_slow, w_crash]).mean() == pytest.approx(1.0, abs=1e-5)
    assert np.concatenate([w_fast, w_slow, w_crash]).max() <= rcfg.max_weight


def test_weights_are_stored_and_weight_the_loss(setup, tmp_path):
    cfg, _, ep = setup
    ep.meta.update(episode_id="e1", map_uid="u1", map_name="m", source="rl:r1")
    ep.weights = np.zeros(len(ep.actions), np.float32)
    ep.weights[: len(ep.actions) // 2] = 2.0
    back = load_episode(save_episode(ep, tmp_path))
    assert np.array_equal(back.weights, ep.weights)

    early = build_window(back, 0, cfg.data)
    late = build_window(back, len(back.frames) - 1, cfg.data)
    assert early["target_weight"][-1, 0] == 2.0 and late["target_weight"].max() == 0.0
    plain = build_window(ep.__class__(**{**ep.__dict__, "weights": None}), 0, cfg.data)
    assert plain["target_weight"][-1].min() == 1.0

    batch = {k: torch.from_numpy(np.asarray(v))[None] for k, v in late.items()}
    k, c = late["target_valid"].shape
    outputs = {
        "steer_logits": torch.randn(1, k, c, cfg.model.steer_bins),
        "steer": torch.randn(1, k, c),
        "gas": torch.randn(1, k, c),
        "brake": torch.randn(1, k, c),
    }
    loss, _ = bc_loss(outputs, batch, cfg.model, cfg.train)
    assert float(loss) == 0.0  # zero-weight steps teach nothing


def test_source_weight_exact_source_before_kind():
    weights = {"dagger": 0.0, "dagger:d1": 10.0}
    assert source_weight("dagger:d1", weights) == 10.0
    assert source_weight("dagger:d2", weights) == 0.0
    assert source_weight("file:x.Replay.Gbx", weights) == 1.0
