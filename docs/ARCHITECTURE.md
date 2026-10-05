# Architecture

Source of requirements: `docs/PLAN.md` (German, from the user). This file is
the binding technical spec for implementers. Shared types live in
`tmagent/interfaces.py`. Any change to a contract below needs an entry in
`docs/decisions.md`.

Game: **TrackMania Nations Forever (TMNF)**, controlled through
**TMInterface** (see `docs/research.md` for verified facts and sources).

## Repository layout

```
tmagent/
  interfaces.py        shared types (Action, GameState, Frame, Episode, game protocols)
  config.py            YAML -> dataclass config, CLI overrides, validation
  experiment.py        experiments/<date>-<name>/ dirs: config, seed, git hash, metrics
  data/
    timeline.py        input events -> per-tick input timeline -> control-rate actions
    episode_io.py      Episode <-> .npz, index.jsonl
    split.py           deterministic split by map_uid
    dataset.py         torch Dataset producing training windows (batch format below)
    augment.py         action-history dropout / noise, image augmentation
    quality.py         QA checks (timestamps, sync, duplicates)
  model/
    encoders.py        frame encoders: tiny_cnn, timm, hf_siglip
    policy.py          temporal causal transformer + action head
    losses.py
    streaming.py       StreamingPolicy: per-step frame-embedding cache for live use
  train/
    train_bc.py        behavior cloning loop (CLI entry)
  runtime/
    scheduler.py       ActionScheduler: time-stamped chunks -> action at any time
    control_loop.py    fixed-rate control thread (never waits on the model)
    inference.py       async inference worker
    profiler.py        per-component latency, p50/p99, deadline misses
    live.py            CLI entry: live play
  eval/
    progress.py        reference polyline, progress %, off-track detection
    harness.py         closed-loop evaluation over maps x seeds (CLI entry)
  game/
    fake.py            FakeGame: tiny 2D racer implementing SyncGame + RealtimeGame
    tmnf/              TMNF bridge via TMInterface (Windows)
tools/
  render_replays.py    replay files -> Episodes (drives game with replay inputs, grabs frames)
  fake_pipeline.py     end-to-end smoke test on FakeGame (no real game needed)
  system_report.py     Phase 0.1: GPU / VRAM / CUDA report -> docs/system.md
  measure_latency.py   Phase 0.4: capture -> preprocess -> model -> input latency -> docs/latency.md
configs/               YAML configs
experiments/           run outputs (git-ignored except README)
docs/                  PLAN.md, ARCHITECTURE.md, WORKLOG.md, decisions.md, research.md, setup_windows.md
tests/                 pytest, CPU-only, no game, no network, no downloads
```

## Timing

- Race time in integer ms. Physics tick `PHYSICS_TICK_MS = 10`.
- `control_hz` (default 60, from the plan) and `frame_hz` (default 20).
  `control_hz % frame_hz == 0` is required; `R = control_hz // frame_hz`
  control steps per frame step.
- Control step `i` is at `round(i * 1000 / control_hz)` ms. Its action is the
  replay input held at that time (sample-and-hold of the 10 ms tick timeline).
  Note: 60 Hz does not divide the 100 Hz physics rate; see decisions.md D-003.
- Frame `k` is at `round(k * 1000 / frame_hz)` ms and shows the state at that
  time before the action at that time is applied.

## Episode storage (format 2, D-007)

- One directory per episode: `<data_root>/episodes/<map_uid>/<episode_id>/`
  - `frames.bin`: per-frame `zlib` blobs (uint8 H x W x C), random access via
    offsets, so a training item decodes only its K frames.
  - `arrays.npz`: `frame_offsets`, `frame_times_ms`, `actions`,
    `action_times_ms`, `positions`, `speeds_kmh`, `frame_shape`.
  - `meta.json`: `EPISODE_META_KEYS` + optional keys (`interfaces.py`).
- `<data_root>/index.jsonl`: one line per episode = meta + `path` (relative
  episode dir) + `num_frames`. Last line per path wins.
- `<data_root>/refs/<map_uid>.npy`: reference polyline (positions of the best
  finished run) for progress evaluation.
- Frames: low resolution, default 128x96 RGB (configurable, also grayscale).

## Training batch format (dataset -> model)

For a window ending at frame step `k_end` with `K` frame steps (`K = 1` is
the single-frame baseline):

| key            | dtype / shape            | meaning |
|----------------|--------------------------|---------|
| `frames`       | uint8 `[B, K, C, H, W]`  | frames at t_k, oldest first, last = current |
| `frame_valid`  | bool `[B, K]`            | False for steps before episode start (zero-padded) |
| `hist_actions` | float32 `[B, K, R, 3]`   | for step k: the R control actions at times in `[t_k - 1000/frame_hz, t_k)` |
| `hist_valid`   | bool `[B, K]`            | False if that history is before episode start, or dropped by augmentation |
| `target`       | float32 `[B, K, C_len, 3]` | for step k: actions at `t_k + j * 1000/control_hz`, j = 0..C_len-1 |
| `target_valid` | bool `[B, K, C_len]`     | False past episode end |
| `progress`     | float32 `[B, K]`         | optional, track progress in [0, 1] (aux / analysis only) |

Loss is computed at every step k (dense, causal), masked by `target_valid`.

## Model (`tmagent/model`)

Sequence per step k: `[a_k] [f_k,1 .. f_k,P]` where `a_k` is one token
embedding the R history actions (MLP over `R*3` values, or a learned
`[NO_ACTION]` embedding when `hist_valid` is False or history is disabled),
and `f_k,*` are `P = tokens_per_frame` tokens from the frame encoder
(patch tokens, adaptive-pooled to P, linear-projected to `d_model`, learned
2D patch position embedding). Learned type embeddings (action/frame).

Time is encoded only as a learned **relative step-distance attention bias**
per head (`rel_bias[n_heads, max_steps]`, index `step_q - step_k`), no
absolute time embedding (D-011). Block-causal: tokens of step k attend to all
tokens of steps <= k. Invalid steps (`frame_valid` False, i.e. before episode
start) hold learned PAD tokens and are key-masked for valid steps (they attend
only to themselves, so no row is fully masked). Consequence, enforced by
tests: the output at a valid step depends only on valid steps <= k and their
relative distances, so dense supervision at every step matches inference.
Custom pre-norm blocks with `F.scaled_dot_product_attention`. Readout: last
frame token of each step -> MLP -> head.

Outputs of `TMPolicy.forward` (all `[B, K, chunk_len, ...]`): regression head
`steer` (pre-tanh), `gas` (logit), `brake` (logit); discrete head
`steer_logits [..., steer_bins]` instead of `steer`. `decode(outputs,
binarize=True) -> [B, K, chunk_len, 3]` in the interfaces.py action layout
(steer tanh / bin expectation, gas/brake sigmoid thresholded at 0.5).

`bc_loss` metrics: `loss`, `loss_steer`, `loss_gas`, `loss_brake`,
`steer_mae`, `gas_acc`, `brake_acc` and `*_step0` variants (chunk row 0).
`train_bc` logs them as `train/<name>` (+ `grad_norm`, `lr`, `s_per_step`) and
`val/<name>`.

Encoders: `tiny_cnn` (fast, tests and first baselines), `timm:<name>`,
`hf:<repo_id>` (e.g. a SigLIP2 checkpoint). External weights load only when
configured; tests never download (timm/hf paths are tested against stubs only).

`StreamingPolicy` (live, eval): keeps the last K encoded frames (each frame
encoded once) and per-step action histories; `observe(image, past_actions)`
then `predict() -> [chunk_len, 3]`; runs the temporal transformer over the
observed steps per call (no KV cache, D-006). `load_streaming_policy(ckpt)`
attaches the checkpoint Config as `.cfg`; `runtime/live.py` adopts the
checkpoint's data section (rates, resolution, history, chunk) and warns on
differences.

Known limit: the attention bias is materialized as `[B, H, L, L]` (~250 MB
fp32 at K=41, P=16, B=32). Needs a blockwise or step-level formulation before
8-30 s contexts (Phase 3).

## Runtime (`tmagent/runtime`)

- `ActionScheduler.publish(chunk, t0_wall)` (row spacing 1/control_hz): chunk of C_len actions whose
  first action applies at wall time `t0_wall`. `action_at(t_wall)` returns the
  linearly interpolated steer and sample-and-hold gas/brake from the newest
  chunk covering `t_wall`; past the chunk end it holds the last action for
  `hold_s` then returns neutral. Thread-safe, lock held only for pointer swap.
- `ControlLoop` runs at `control_hz` on its own thread, uses an absolute
  schedule (`perf_counter`, sleep + short spin), calls
  `game.set_action(scheduler.action_at(now))`, never touches the model.
  Deadline miss = the action for slot n was sent after slot n+1 started.
- `InferenceWorker` loop: take `game.latest_frame()`, if new: append to
  `StreamingPolicy`, `predict`, publish chunk with `t0_wall = frame.wall_time`
  (the chunk starts at the observation time; stale leading actions are simply
  in the past and skipped by `action_at`).
- `LatencyProfiler`: named sections, p50/p95/p99/max, deadline miss %, dump
  to JSON and Markdown.

## Evaluation (`tmagent/eval`)

- Reference polyline per map = positions of a reference run (from rendering).
  Progress = arc length of the projection of the car position onto the
  polyline (searching only a window ahead of the last projection) / total length.
- Off-track proxy: distance to polyline > threshold, or speed < threshold for
  > N s (stuck). Episode ends on finish, timeout, or stuck.
- Harness runs `maps x seeds` with a `SyncGame` (deterministic, any speed) or
  `RealtimeGame` (true live timing), writes per-episode JSONL and a summary:
  finish rate, median progress, time vs reference, deadline miss %.

## Experiments

`tmagent.experiment.create_run(name, cfg)` -> `experiments/<YYYY-MM-DD>-<name>/`
with `config.yaml`, `git.txt` (hash + dirty flag), `seed.txt`, `metrics.jsonl`,
`checkpoints/`. Every training/eval CLI uses it.

## Testing rules

- `pytest` must pass on CPU with no game, no GPU, no network.
- `tools/fake_pipeline.py` exercises the full chain on FakeGame: scripted
  expert runs -> render episodes -> dataset -> train a tiny model for a few
  steps -> closed-loop eval -> live loop for a few seconds.

## TMNF bridge (`tmagent/game/tmnf`)

Facts and sources: `docs/research.md`. Decisions D-008..D-010.

- `plugin/TMAgentLink.as`: our TMInterface 2.x AngelScript plugin, TCP server
  on `game.tmi_port`. Protocol spec: `tmagent/game/tmnf/PROTOCOL.md`.
- `client.py`: Python socket client (framing, reader thread, demux).
- `game.py`: `TMNFGame(game_cfg, data_cfg)` implementing `SyncGame` and
  `RealtimeGame` on top of the client.
  - Sync (rendering, eval): game paused (`SetSpeed(0)`) between commands;
    `step(action, n)` runs n ticks then pauses; `grab_frame()` captures via
    `Graphics::CaptureScreenshot(vec2(W, H))` in `Render()` while paused, so
    the image shows exactly the current tick.
  - Realtime (live): game at `game_speed`; the plugin applies the latest
    received input every tick; frames pushed at the configured size.
  - Binary steering by default (`steer_mode`), see D-010.
- `replay.py`: `.Replay.Gbx` -> normalized events -> `InputTimeline` (pygbx,
  optional) and TMI input-script text -> `InputTimeline` (no GPL code).
- `tests/tmnf/`: protocol tests against a Python fake plugin server that
  speaks the same protocol (the real game cannot run in CI).
