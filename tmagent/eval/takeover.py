"""Interactive drives: the policy drives in sync mode, a human can take over and hand back.

Phases (TakeoverControl.phase):
  "agent"    sync mode, the policy is stepped tick by tick (as in harness.run_episode_sync)
  "player"   realtime mode with plugin passthrough: the player's keyboard drives at game speed
  "waiting"  the agent got stuck / went off track; the game is paused until the player takes
             over or lets the agent retry
The player takes over with an arrow key (or a request) and hands back with H (or a request).
Every player segment is recorded (frames, keys, positions) and can be saved as a training
episode: these are the recoveries plain behaviour cloning never sees.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import Config
from tmagent.data.player import (
    KeyReader,
    PlayerSegment,
    Requests,
    drive_segment,
    read_keyboard,
    store_segment,
)
from tmagent.eval.harness import _episode_result, drive_frame_period, policy_rows
from tmagent.eval.progress import ProgressTracker, ReferencePath
from tmagent.interfaces import ChunkPolicy, GameState

WAIT_POLL_S = 0.05


class TakeoverControl(Requests):
    """Requests from a UI: "take_over", "hand_back", "retry"; phase agent / player / waiting."""

    def __init__(self) -> None:
        super().__init__("agent")


def save_segment(
    seg: PlayerSegment, cfg: Config, map_uid: str, map_name: str, ckpt: str
) -> Path | None:
    """Store a takeover segment as an episode (source "takeover"); None if too short."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "episode_id": f"{map_uid}-takeover-{stamp}-{seg.start_ms}",
        "map_uid": map_uid,
        "map_name": map_name,
        "source": f"takeover:{stamp}",
        "renderer": "takeover",
        "agent_ckpt": ckpt,
    }
    return store_segment(seg, cfg, meta)


def _drive_player(
    game: Any,
    policy: ChunkPolicy,
    cfg: Config,
    tracker: ProgressTracker,
    control: TakeoverControl,
    state: GameState,
    read_keys: KeyReader,
    on_frame: Callable[[ProgressTracker], None] | None,
) -> tuple[GameState, PlayerSegment]:
    """Realtime + passthrough until hand back (H / request) or the finish; the policy keeps
    observing the frames so its history is current when it drives again."""
    d, r = cfg.data, cfg.data.actions_per_frame
    control.phase = "player"
    game.hand_to_player()
    seg = PlayerSegment(start_ms=int(state.race_time_ms))
    frame_ms = 1000.0 / d.frame_hz
    next_obs = [float(state.race_time_ms)]

    def observe(frame: Any) -> None:
        if frame.race_time_ms >= next_obs[0]:
            policy.observe(frame.image, np.tile(seg.last_action(), (r, 1)))
            next_obs[0] += frame_ms

    def stop(st: GameState, keys: tuple) -> str | None:
        if on_frame is not None:
            on_frame(tracker)
        tracker.update(st, judge=False)
        if tracker.done:
            seg.finished = tracker.reason == "finished"
            return "finished"
        if keys[4] or control.pop() == "hand_back":
            return "hand_back"
        return None

    drive_segment(game, seg, read_keys, stop, on_frame=observe)
    state = game.take_back()
    tracker.revive()
    control.phase = "agent"
    return state, seg


def run_takeover_episode(
    game: Any,
    policy: ChunkPolicy,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
    control: TakeoverControl,
    on_frame: Callable[[ProgressTracker], None] | None = None,
    on_segment: Callable[[PlayerSegment], None] | None = None,
    read_keys: KeyReader = read_keyboard,
) -> dict:
    """One interactive episode; the game needs hand_to_player() / take_back() (TMNFGame).

    Ends at the finish, at a timeout, or when on_frame raises. Stuck / offtrack do not end
    it: the game pauses in phase "waiting" until a takeover (arrow key / request) or a
    "retry" request.
    """
    d = cfg.data
    rng = np.random.default_rng(seed)
    policy.reset()
    control.phase = "agent"
    game.load_map(map_ref)
    state = game.start_race()
    tracker = ProgressTracker(ref, cfg.eval)
    tracker.update(state)
    past = np.zeros((d.actions_per_frame, 3), np.float32)
    k = 0
    takeovers, player_ms = 0, 0
    while not tracker.done:
        if on_frame is not None:
            on_frame(tracker)
        req = control.pop()
        if req == "take_over" or any(read_keys()[:4]):
            state, seg = _drive_player(
                game, policy, cfg, tracker, control, state, read_keys, on_frame
            )
            takeovers += 1
            player_ms += seg.duration_ms
            if on_segment is not None:
                on_segment(seg)
            past = np.tile(seg.last_action(), (d.actions_per_frame, 1))
            k = int(state.race_time_ms) * d.frame_hz // 1000
            continue
        if control.phase == "waiting":
            if req == "retry":
                control.phase = "agent"
            else:
                time.sleep(WAIT_POLL_S)
            continue
        frame = game.grab_frame()
        policy.observe(frame.image, past)
        rows = policy_rows(policy, cfg, rng)
        state = drive_frame_period(game, rows, state, k, cfg, tracker)
        past = rows
        k += 1
        if tracker.done and tracker.reason in ("stuck", "offtrack"):
            tracker.revive()
            control.phase = "waiting"
    control.phase = "agent"
    result = _episode_result(map_ref, seed, "takeover", tracker, ref)
    return {**result, "frames": k, "takeovers": takeovers, "player_ms": player_ms}
