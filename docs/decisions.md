# Decisions

Format: ID, date, decision, reason, status. Record what was tried, measured,
and rejected. Numbers only from own measurements.

## D-001 (2026-10-05) Game = TrackMania Nations Forever + TMInterface
- Decision: target TMNF (free) driven through TMInterface (TMI) instead of
  TM2020/Openplanet.
- Reason: user request; TMNF physics is deterministic at fixed 10 ms ticks, and
  TMI exposes tick-level input injection, state reads and frame capture, so
  replay inputs can be re-driven and frames rendered in exact sync. Details
  and sources: `docs/research.md`.
- Status: accepted.

## D-002 (2026-10-05) Build-only session, no training or data generation here
- Decision: this cloud session builds and verifies code only (CPU tests,
  FakeGame end-to-end). No replay downloads, no clip rendering, no real
  training. Those run on the user's Windows machine.
- Status: accepted (user instruction).

## D-003 (2026-10-05) Control rate 60 Hz vs 100 Hz physics
- Fact: TMNF simulates at 100 Hz (10 ms ticks). 60 Hz does not divide 100 Hz,
  so a 60 Hz control loop changes inputs on a 10/20/20 ms tick pattern.
- Decision: keep the plan's 60 Hz as default (`data.control_hz =
  runtime.control_hz = 60`), all code takes the rate from config.
- Open: 50 Hz (every 2nd tick) aligns exactly with physics and is the
  recommended alternative. Decide by closed-loop metrics in Phase 2.
- Status: open.

## D-004 (2026-10-05) One action token per frame step
- Decision: interleave one action token per frame step (embedding the
  `control_hz / frame_hz` control actions since the previous frame), instead
  of one token per control step. Keeps the sequence length at
  `K * (tokens_per_frame + 1)`.
- Status: accepted.

## D-005 (2026-10-05) Dense causal supervision
- Decision: predict a chunk at every frame step of a window, loss over all
  steps (masked). Equivalent to K samples per window at the cost of one.
- Status: accepted.

## D-006 (2026-10-05) No KV-cache in the first streaming policy
- Decision: `StreamingPolicy` caches per-frame encoder outputs and recomputes
  the small temporal transformer per call. KV-cache streaming only if the
  Phase 0/4 latency measurement shows the transformer is the bottleneck.
- Status: accepted.

## D-007 (2026-10-05) Episode storage = one compressed .npz per run + index.jsonl
- Decision: simple per-episode `.npz` files with low-res frames (default
  128x96 RGB at 20 fps) instead of tar shards. Reconsider if file count or
  read throughput becomes a problem.
- Status: accepted.

## D-008 (2026-10-05) GPL tools stay optional
- pygbx (GPL-3) and the `tminterface` pip package (GPL-3, TMI 1.x only) are
  not required dependencies. pygbx is an optional extra used only inside
  `tmagent/game/tmnf/replay.py`; the TMI input-script path (`dump_inputs`
  text) needs no GPL code. The repository has no license yet; the user decides.
- Status: accepted.

## D-009 (2026-10-05) Own TMI 2.x plugin instead of reusing Linesight code
- Linesight has no LICENSE file (MIT only declared in setup.py). tmagent
  writes its own AngelScript plugin (`TMAgentLink.as`) and Python client with
  its own protocol, using only the TMI plugin API calls listed in
  research.md. API names marked UNVERIFIED are collected in one place in the
  plugin and must be checked against the installed TMI version (Phase 0).
- Status: accepted.

## D-010 (2026-10-05) Binary steering by default on TMNF
- Most TMX replays come from keyboard drivers (binary steer events), and
  `SetInputState(InputType::Left/Right)` is verified, while analog steer
  injection (`InputType::Steer`) is only snippet-verified and in TMI 1.x
  needed a detected gamepad. `game.steer_mode = binary` (threshold 0.5);
  `analog` available once verified on the user's machine.
- Status: accepted, revisit in Phase 0.
