"""Reward-weighted fine-tuning data (a small RL test, AWR style).

The policy drives with sampled actions (steer class from the softmax, gas / brake from their
Bernoulli, both at `temperature`), and the executed actions are the labels. Every control step
gets a reward: progress along the reference line, a penalty when the run ends stuck / offtrack /
stalled, a bonus at the finish. The advantage of a step is its discounted return over a few
seconds minus the mean return of all attempts on the same map at the same place on the line,
and the training weight is exp(advantage / beta), so behaviour that went better than usual at
that spot is imitated more. Human data stays in the mix and keeps the policy close to BC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tmagent.config import Config
from tmagent.data.timeline import control_times_ms, frame_times_ms
from tmagent.eval.harness import control_row_for_tick
from tmagent.eval.progress import ProgressTracker, ReferencePath
from tmagent.interfaces import PHYSICS_TICK_MS, Action, Episode, SyncGame
from tmagent.model.policy import steer_bin_centers

CRASH_REASONS = ("stuck", "offtrack", "stall")


@dataclass
class RLConfig:
    temperature: float = 1.0  # softmax / sigmoid temperature of the sampled actions
    stall_ms: int = 4000  # end the attempt after this long without progress along the line
    stall_m: float = 1.0
    max_step_m: float = 5.0  # per-step progress is clipped: projection jumps are no reward
    crash_penalty_m: float = 50.0  # stuck / offtrack / stall end, in metres of progress
    finish_bonus_m: float = 50.0
    horizon_s: float = 3.0  # discount horizon of the return
    bin_m: float = 5.0  # baseline: mean return of all attempts within this stretch of line
    beta: float = 1.0  # weight = exp(normalized advantage / beta)
    max_weight: float = 20.0


@dataclass
class Attempt:
    """One collected attempt; `progress_m` is the line position at every control step."""

    episode: Episode
    progress_m: np.ndarray
    reason: str
    progress: float
    seconds: float


def sample_rows(
    outputs: dict[str, np.ndarray],
    r: int,
    rng: np.random.Generator,
    temperature: float,
) -> np.ndarray:
    """R executable rows (steer in the bin centers, gas / brake 0 or 1) sampled from the
    first R chunk rows of the policy's raw outputs (needs the discrete steer head)."""
    if "steer_logits" not in outputs:
        raise ValueError("RL sampling needs a model with model.head=discrete")
    t = max(temperature, 1e-6)
    logits = outputs["steer_logits"][:r] / t
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    centers = steer_bin_centers(logits.shape[-1]).numpy()
    steer = np.array([centers[rng.choice(len(row), p=row)] for row in p], np.float32)
    gas = rng.random(r) < 1.0 / (1.0 + np.exp(-outputs["gas"][:r] / t))
    brake = rng.random(r) < 1.0 / (1.0 + np.exp(-outputs["brake"][:r] / t))
    return np.stack([steer, gas, brake], axis=1).astype(np.float32)


def collect_attempt(
    game: SyncGame,
    policy,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
    rcfg: RLConfig,
) -> Attempt:
    """One attempt with sampled actions; labels are the executed actions."""
    d = cfg.data
    r = d.actions_per_frame
    rng = np.random.default_rng(seed)
    policy.reset()
    game.load_map(map_ref)
    state = game.start_race()
    tracker = ProgressTracker(ref, cfg.eval)
    tracker.update(state)

    frames, actions, positions, speeds, line = [], [], [], [], []
    past = np.zeros((r, 3), np.float32)
    best_s, best_t, ticks, k = 0.0, 0, 0, 0
    while not tracker.done:
        frame = game.grab_frame()
        frames.append(frame.image)
        policy.observe(frame.image, past)
        rows = sample_rows(policy.predict_outputs(), r, rng, rcfg.temperature)

        t_next = round((k + 1) * 1000 / d.frame_hz)
        while state.race_time_ms < t_next and not tracker.done:
            t = int(state.race_time_ms)
            while (
                round(len(actions) * 1000 / d.control_hz) // PHYSICS_TICK_MS * PHYSICS_TICK_MS <= t
            ):
                actions.append(rows[min(max(len(actions) - k * r, 0), r - 1)])
                positions.append(np.asarray(state.position, np.float32).reshape(3))
                speeds.append(float(state.speed_kmh))
                line.append(tracker.s)

            j = control_row_for_tick(t, k, d.control_hz, r)
            state = game.step(Action.from_array(rows[j]), 1)
            ticks += 1
            tracker.update(state)
            if tracker.max_s > best_s + rcfg.stall_m:
                best_s, best_t = tracker.max_s, int(state.race_time_ms)
            elif state.race_time_ms - best_t > rcfg.stall_ms:
                tracker.finish("stall")

        past = rows
        k += 1

    n_f = len(frames)
    n_a = max(len(actions), (n_f - 1) * r + 1)
    while len(actions) < n_a:
        actions.append(actions[-1])
        positions.append(positions[-1])
        speeds.append(speeds[-1])
        line.append(line[-1])

    res = tracker.result()
    meta = {
        "race_time_ms": int(res["race_time_ms"]),
        "finished": bool(res["finished"]),
        "finish_time_ms": res.get("finish_time_ms"),
        "frame_hz": int(d.frame_hz),
        "control_hz": int(d.control_hz),
        "resolution": [int(d.resolution[0]), int(d.resolution[1])],
        "channels": int(d.channels),
        "desync": False,
    }
    episode = Episode(
        frames=np.stack(frames),
        frame_times_ms=frame_times_ms(n_f, d.frame_hz),
        actions=np.stack(actions[:n_a]).astype(np.float32),
        action_times_ms=control_times_ms(n_a, d.control_hz),
        positions=np.stack(positions[:n_a]),
        speeds_kmh=np.array(speeds[:n_a], np.float32),
        meta=meta,
    )
    return Attempt(
        episode=episode,
        progress_m=np.array(line[:n_a], np.float64),
        reason=str(res["reason"]),
        progress=float(res["progress"]),
        seconds=ticks * PHYSICS_TICK_MS / 1000,
    )


def step_rewards(progress_m: np.ndarray, reason: str, rcfg: RLConfig) -> np.ndarray:
    """Reward per control step: clipped progress along the line, plus the end bonus / penalty
    on the last step."""
    gain = np.diff(progress_m, append=progress_m[-1])
    rewards = np.clip(gain, -rcfg.max_step_m, rcfg.max_step_m)
    if reason == "finished":
        rewards[-1] += rcfg.finish_bonus_m
    elif reason in CRASH_REASONS:
        rewards[-1] -= rcfg.crash_penalty_m
    return rewards


def discounted_returns(rewards: np.ndarray, control_hz: int, horizon_s: float) -> np.ndarray:
    gamma = 1.0 - 1.0 / (horizon_s * control_hz)
    out = np.zeros(len(rewards), np.float64)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def advantage_weights(attempts: list[Attempt], control_hz: int, rcfg: RLConfig) -> list[np.ndarray]:
    """Per-step training weights for all attempts on one map (mean 1 over the map)."""
    returns = [
        discounted_returns(step_rewards(a.progress_m, a.reason, rcfg), control_hz, rcfg.horizon_s)
        for a in attempts
    ]
    bins = [np.floor(a.progress_m / rcfg.bin_m).astype(np.int64) for a in attempts]
    all_bins, all_returns = np.concatenate(bins), np.concatenate(returns)

    baseline: dict[int, float] = {}
    for b in np.unique(all_bins):
        baseline[int(b)] = float(all_returns[all_bins == b].mean())

    advantages = [
        g - np.array([baseline[int(b)] for b in bs]) for g, bs in zip(returns, bins, strict=True)
    ]
    scale = float(np.concatenate(advantages).std()) + 1e-6
    weights = [np.minimum(np.exp(a / scale / rcfg.beta), rcfg.max_weight) for a in advantages]
    mean = float(np.concatenate(weights).mean())
    return [(w / mean).astype(np.float32) for w in weights]
