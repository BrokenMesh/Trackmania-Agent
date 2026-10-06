"""DAgger data collection: the policy drives, the LineExpert labels every control step.

Control alternates in random blocks: the agent drives `agent_s` seconds, then the expert
`expert_s` seconds. On top, the expert takes over when the car is more than `intervene_m` off
the reference line or crawls below `slow_kmh`, for at least `hold_s` and until it is back
within `release_m`. Labels are the expert's action at every control step, whoever drove: the
states the agent visits get corrective targets, and the expert blocks carry the run along the
whole track (VERIFIED need: with interventions only, the agent was already in a wall when the
expert took over and runs ended at 1-7 % progress).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tmagent.config import Config
from tmagent.data.timeline import control_times_ms, frame_times_ms
from tmagent.eval.expert import LineExpert
from tmagent.eval.harness import control_row_for_tick, policy_rows
from tmagent.eval.progress import ProgressTracker, ReferencePath
from tmagent.interfaces import PHYSICS_TICK_MS, Action, ChunkPolicy, Episode, SyncGame


@dataclass
class DaggerConfig:
    intervene_m: float = 2.5
    release_m: float = 1.5
    agent_s: tuple[float, float] = (0.5, 2.0)  # random block lengths (uniform)
    expert_s: tuple[float, float] = (1.0, 2.0)
    hold_s: float = 1.0
    slow_kmh: float = 10.0
    slow_after_ms: int = 3000
    stall_ms: int = 4000  # end the episode after this long without progress along the line
    stall_m: float = 1.0


def collect_episode(
    game: SyncGame,
    policy: ChunkPolicy,
    expert: LineExpert,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
    dcfg: DaggerConfig | None = None,
) -> tuple[Episode, dict]:
    """One mixed-control episode with expert labels; returns (episode, stats)."""
    dcfg = dcfg or DaggerConfig()
    d = cfg.data
    r = d.actions_per_frame
    rng = np.random.default_rng(seed)
    policy.reset()
    expert.reset()
    game.load_map(map_ref)
    state = game.start_race()
    tracker = ProgressTracker(ref, cfg.eval)
    tracker.update(state)
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    speeds: list[float] = []
    past = np.zeros((r, 3), np.float32)
    intervening, hold_until = False, 0
    best_s, best_t = 0.0, 0
    block_expert, block_end = False, int(rng.uniform(*dcfg.agent_s) * 1000)
    expert_ticks = ticks = 0
    k = 0
    while not tracker.done:
        frame = game.grab_frame()
        frames.append(frame.image)
        policy.observe(frame.image, past)
        rows = policy_rows(policy, cfg, rng)
        executed = rows.copy()
        t_next = round((k + 1) * 1000 / d.frame_hz)
        while state.race_time_ms < t_next and not tracker.done:
            t = int(state.race_time_ms)
            label = expert.act(state.position, state.velocity, state.speed_kmh)
            _, dist = expert.nearest(state.position)
            while (
                round(len(labels) * 1000 / d.control_hz) // PHYSICS_TICK_MS * PHYSICS_TICK_MS <= t
            ):
                labels.append(label)
                positions.append(np.asarray(state.position, np.float32).reshape(3))
                speeds.append(float(state.speed_kmh))
            if t >= block_end:
                block_expert = not block_expert
                span = dcfg.expert_s if block_expert else dcfg.agent_s
                block_end = t + int(rng.uniform(*span) * 1000)
            slow = t > dcfg.slow_after_ms and state.speed_kmh < dcfg.slow_kmh
            if not intervening and (dist > dcfg.intervene_m or slow):
                intervening, hold_until = True, t + int(dcfg.hold_s * 1000)
            elif intervening and t >= hold_until and dist < dcfg.release_m and not slow:
                intervening = False
            j = control_row_for_tick(t, k, d.control_hz, r)
            expert_drives = intervening or block_expert
            action = label if expert_drives else rows[j]
            executed[j] = action
            expert_ticks += expert_drives
            ticks += 1
            state = game.step(Action.from_array(action), 1)
            tracker.update(state)
            if tracker.max_s > best_s + dcfg.stall_m:
                best_s, best_t = tracker.max_s, int(state.race_time_ms)
            elif state.race_time_ms - best_t > dcfg.stall_ms:
                tracker.finish("stall")  # wedged: expert labels stop being meaningful
        past = executed
        k += 1
    n_f = len(frames)
    n_a = max(len(labels), (n_f - 1) * r + 1)
    while len(labels) < n_a:
        labels.append(labels[-1])
        positions.append(positions[-1])
        speeds.append(speeds[-1])
    res = tracker.result()
    meta = {
        "race_time_ms": int(res["race_time_ms"]),
        "finished": bool(res["finished"]),
        "frame_hz": int(d.frame_hz),
        "control_hz": int(d.control_hz),
        "resolution": [int(d.resolution[0]), int(d.resolution[1])],
        "channels": int(d.channels),
        "desync": False,
    }
    ep = Episode(
        frames=np.stack(frames),
        frame_times_ms=frame_times_ms(n_f, d.frame_hz),
        actions=np.stack(labels[:n_a]).astype(np.float32),
        action_times_ms=control_times_ms(n_a, d.control_hz),
        positions=np.stack(positions[:n_a]),
        speeds_kmh=np.array(speeds[:n_a], np.float32),
        meta=meta,
    )
    stats = {
        "progress": res["progress"],
        "reason": res["reason"],
        "expert_share": expert_ticks / max(ticks, 1),
        "seconds": ticks * PHYSICS_TICK_MS / 1000,
    }
    return ep, stats
