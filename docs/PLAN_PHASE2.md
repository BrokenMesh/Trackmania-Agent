# Plan, stage 2: memory agent (learning a map across attempts)

Stage 2 starts after PLAN.md Phase 3. That means the closed-loop eval harness exists, the
context model runs with 2-8 s of history, and the single-frame baseline has been measured.
Milestones here are numbered M0-M7 so they do not clash with the phase numbers in PLAN.md.
The working rules in PLAN.md apply unchanged. The most relevant ones: small verifiable steps,
decisions only from closed-loop metrics, and no RL unless the user asks for it.

## Goal

An agent that gets better on an **unseen** map over several attempts by remembering its
earlier attempts, and that ends hopeless runs itself with a restart action.

- **Short-term context** (realtime, about 15 s): the current attempt at frame_hz (20 fps),
  plus the action history if Phase 3 shows that it helps. It is reset at every restart.
- **Long-term memory** (1 frame per second, at most 300 frames = 5 min of driving): the
  earlier attempts on the same map, plus one summary token per attempt (outcome, progress,
  time). It survives restarts and is cleared only when the map changes. When it is full,
  the oldest frames are dropped first (FIFO); summary tokens are kept. Decided by the user,
  see "Decisions".
- **Restart action**: a 4th action output. Pressing it ends the attempt, moves the attempt
  into the memory and restarts the race.

Success criteria. Set the numbers after M1 and enter them here.

- Held-out maps: median progress and finish rate at attempt N are clearly higher than at
  attempt 1 (_TBD_).
- The same checkpoint with the memory blanked does not show this improvement (memory
  ablation). Without this check, an improvement curve proves nothing.
- Restart: precision and recall on lost runs _TBD_. On runs that are still good it is
  almost never pressed (false restart rate _TBD_).

## The central risk: the model ignores the memory

If the label of the current attempt is always the original replay input, the label does
not depend on the memory. A model that ignores the memory reaches the same loss, and
training takes that shortcut. This is the copycat effect from D-013, one level higher.
Design rules that follow from it:

- **R1. The memory must carry information that the short-term context lacks.** The natural
  source is anticipation. Replay drivers know the map: they brake before blind jumps and
  turn in before a corner is visible. On a new map, only earlier attempts can show what
  comes next. Prefer attempt data in which earlier attempts drove past points that the
  current attempt cannot see yet.
- **R2. History-dependent targets (variant B, decided).** The target of attempt k depends
  on attempts 1..k-1: wherever the memory shows no crash, the driver acts **naively** (like
  someone who does not know the map yet); at a spot where an earlier attempt crashed, it
  acts with the replay's correct timing. The same image therefore has two different correct
  targets, and the memory decides which one applies. This is a synthetic "learning history"
  in the style of Algorithm Distillation. Details: step 5 of the session generator below.
- **R2a. Mistakes must be systematic, not random.** With random perturbations, variant B
  degenerates to "target = replay": the fixed spot looks like the replay (which the model
  predicts anyway), and mistakes further ahead were never driven, so they are not in the
  memory and are only label noise. Only a naive behavior that is a fixed function of the
  image ("too late at hard spots") makes the target depend on the memory.
- **R3. Strict split by map**, and training sessions with an empty memory (first attempt).
  Otherwise the model memorizes the training maps and does not need the memory.
- **R4. More maps beats more runs per map.** The memory skill has to be general. Aim for
  several hundred to a few thousand Stadium maps from TMX instead of 65
  (`tools/tmx_download.py --keyboard-only` already works for that).
- **R5. Only multi-attempt closed-loop eval with memory ablation decides.** Training loss
  showed nothing in D-013 and will not here either.

## Context layout and token budget (estimate, check in M0)

| region | content | rate | tokens per step | size |
|---|---|---|---|---|
| short-term | current attempt: frame tokens + action token | 20 fps | 16 + 1 | 15 s = 300 steps = 5100 |
| memory frames | earlier attempts, oldest first | 1 fps | 4-8 (more pooling than short-term) | 300 frames x 8 = 2400 |
| memory summaries | per attempt: outcome, progress at the end, time, reason for the end | 1 per attempt | 1 | about 10-30 |

That is about 7.5k tokens. This is feasible on the RTX 4060 with a small model, but only
with the attention change in M0. The memory is defined by frames, not attempts:
`memory.fps` x `memory.max_frames` = covered driving time (1 fps x 300 = 5 min). At
300 km/h, 1 fps means about 80 m between memory frames and 0.5 fps about 160 m, so a
turn-in point can fall between two frames. M1 compares 1 fps / 300 frames (5 min) with
0.5 fps / 300 frames (10 min).

Positions and attention:

- Short-term: unchanged, relative step bias (D-011), block-causal.
- Memory: its own type embedding, plus an attempt index relative to the current attempt
  (-1, -2, ...), plus time within the attempt or track progress (an M1 ablation decides
  which). Memory tokens attend causally over attempts. The current attempt attends to the
  whole memory, because the memory lies entirely in the past.
- Memory frames are encoded **once** and cached (per frame in training, per attempt at
  runtime), like the per-step cache in `StreamingPolicy`.

## Making sure variant B learns

Learning from memory is only accepted if all of the following checks pass, in this order.
Each one is cheaper than the next, so a broken design fails early.

- **L1. Data check (generator, M3).** For every session: count the ticks where the target
  differs from the target the same image would get with an empty memory ("memory-dependent
  ticks"). Report this per map and in total. If it is close to 0, the data cannot teach
  memory use and training does not start.
- **L2. FakeGame proof (CPU, before any game rendering, M1a).** FakeGame gets maps with a
  blind hard spot (a turn that is visible only shortly before). The same session generator
  runs on it, a tiny model is trained, and the probes of L3 run. This proves the data
  generator, the model wiring and the training loop end to end, and it becomes a pytest
  regression test (CPU only).
- **L3. Offline probes (every M4 eval, logged as `val/mem_*`).** Measured **only on the
  memory-dependent ticks**, because the overall loss is dominated by ticks that do not
  need the memory and hides the effect:
  - `mem_blank_gap`: loss with the memory blanked minus loss with the real memory. It must
    be clearly positive.
  - `mem_swap_follow`: the same current window with a counterfactual memory (crash at this
    spot vs. no crash at this spot). The fraction of predictions that switch between the
    correct and the naive timing as the memory says. Target: close to 1.
  - `mem_foreign`: memory from another map. Predictions must behave like blank memory
    (no spurious use).
- **L4. Sampling.** Memory-dependent ticks are rare (a few hard spots per minute).
  Training oversamples windows around them (config `memory.hard_spot_oversample`) so that
  the gradient sees them; L3 and the normal driving metrics make sure this does not break
  the rest of the driving.
- **L5. Closed loop (M5).** On held-out maps the improvement curve over the attempts must
  exist with memory and disappear with the memory blanked. The fail points must move
  along the track from attempt to attempt, as in the training sessions.

## Changes per module

| module | change |
|---|---|
| `interfaces.py` | `ACTION_DIM` 3 -> 4 (`restart` in [0, 1]); `Action.restart`; new `Session` type (map_uid, ordered list of attempts = episode refs + outcome) |
| `config.py` | new section `memory`: `enabled`, `fps` (1), `max_frames` (300, FIFO), `tokens_per_frame`, `summary_tokens`, `position` (time or progress); `data.history_s` up to 15; `train.loss_weights.restart` |
| `data/timeline.py` | naive operator (delay or drop an onset by Δ) on the tick timeline (M3) |
| `data/hard_spots.py` (new) | hard-spot detection on a clean run (onset at high speed before curvature, a speed drop or a jump); memory-dependent tick count (L1) |
| `game/fake.py` | maps with a blind hard spot for the L2 proof |
| `data/session_io.py` (new) | `sessions.jsonl`: a session refers to existing episodes, so frames are not copied; the memory is subsampled to `memory.fps` at load time |
| `data/dataset.py` | session sampler: draw a map, a session and a current attempt; memory = all earlier attempts (sometimes empty, R3); window in the current attempt as now |
| `data/augment.py` | memory dropout (blank or drop whole attempts) so that "no memory" stays a valid case |
| `model/policy.py` | memory region, attention mask current->memory, summary token embedding, restart head (logit); no materialized `[B,H,L,L]` bias (M0) |
| `model/losses.py` | restart BCE with class weight (restart is rare), metrics `restart_precision/recall` |
| `model/streaming.py` | memory buffer: on restart, subsample the finished attempt to `memory.fps`, encode it, append it (drop the oldest frames beyond `max_frames`), write the summary token, clear the short-term context |
| `runtime/` | restart output above a threshold -> `game.restart()` with a cooldown (no double restarts); the attempt boundary goes to the policy |
| `eval/harness.py` | multi-attempt mode: per map N attempts or a time budget; metrics per attempt index; memory ablation as a flag; restart statistics |
| `eval/progress.py` | reuse "lost" detection (off-track, stuck, no progress) as the source of the restart label |
| `game/tmnf` plugin + client | `SAVE_STATE` / `LOAD_STATE` at any tick (TMI `SaveState` / `RewindToState` are already used for t=0), for prefix reuse in M3 |
| `tools/render_sessions.py` (new) | creates attempt sessions from replays (M3) |
| `tools/episode_gif.py` | also show restart output and memory attempts |

## Generating attempt sessions (M3)

1. **Source**: keyboard replays per map (more maps, R4).
2. **Hard spots and the naive operator** (variant B):
   - Hard spots per map, from the clean run: places where the replay driver acts before
     the reason is visible. In practice: steer or brake onsets at high speed, before strong
     curvature, a speed drop, or a jump. Thresholds are config values.
   - Naive operator: at a hard spot, the onset comes **too late by Δ** (Δ from a fixed
     range, for example 200-400 ms, drawn once per spot and map so that it is the same in
     every attempt). Other forms of the same idea: a brake phase left out, a press
     shortened. What matters is that the mistake is the same every time the driver does
     not know the spot yet.
   - Spots where the naive version does not lead to a fail within 15 s are not hard spots
     for this map. They stay naive and are not counted.
3. **Render only after the perturbation**. Up to the perturbation tick the attempt is
   identical to the clean episode. Reuse its frames, `LOAD_STATE` at the perturbation tick,
   and render only the rest. This saves most of the game time. The clean run's states are
   saved every N ticks during the normal render.
4. **Detect the failure (user decision)**: an attempt is lost when it **diverges clearly
   from the successful source run**: position distance to the source run at the same
   track progress above a threshold, or progress falling behind the source run by more
   than a threshold. Both thresholds are config values, set in M2 from the data. The
   attempt ends at divergence + reaction delay (random 0.3-1.5 s), and from there the
   restart label is 1. **It ends at the latest 15 s after the input change.** A perturbed
   attempt that has not diverged after 15 s is not a fail; it is dropped (or kept as a
   successful attempt if it finishes).
5. **Assemble the session (variant B)**: attempts 1..n on one map. Every hard spot has a
   knowledge state: unknown (naive) or known (correct timing).
   - Attempt 1: all spots unknown. It crashes at the first hard spot that leads to a fail.
   - Attempt k: the spot where attempt k-1 crashed becomes known; everything else is like
     attempt k-1. So attempt k crashes at the next hard spot, and the fails move along the
     track until an attempt finishes.
   - Target of attempt k = its own inputs (correct timing at known spots, naive at unknown
     ones) plus the restart label after its fail.
   - Mix in sessions with an empty memory (attempt 1) and sessions where the memory was
     cut short by the FIFO limit, so that "spot no longer in memory" means naive again.
6. **Cost**: rendering time is roughly (number of maps) x (sessions per map) x (attempts)
   x (average segment after the perturbation) / (render speed). Measure it before M3 at
   scale. The render speed-ups (merge step and grab into one command, several game
   instances) belong in M0.

## Milestones

### M0: Prerequisites
- PLAN.md Phases 2 and 3 done: harness, baseline numbers, context model up to 8 s.
- Attention without a materialized bias (blockwise or step-level bias, so that SDPA/flash
  works). Test: same outputs as today for short contexts.
- Render speed: one STEP_AND_GRAB command, then several game instances (one port each,
  replays split across them).
- Map scaling: download several hundred maps with keyboard replays. Restart the game after
  adding maps (`docs/setup_windows.md`).
- Exit: 15 s short-term context trains on the 4060 and is measured closed-loop.

### M1: Prove the memory signal
- **M1a, FakeGame proof (L2)**: session generator + tiny model + L3 probes on FakeGame,
  as a pytest regression test. Exit: `mem_swap_follow` close to 1 and `mem_blank_gap`
  clearly positive. Without this, nothing is rendered in the real game.
- **M1b, oracle memory on TMNF**:
- Memory = **one clean run of the same map** at 1 fps (a replay, not an agent attempt).
- Compare on held-out maps: progress and finish rate with and without this memory.
  Variants: 1 vs 2 fps; position encoding time vs progress.
- Exit: a clear gain with memory. **If there is none, stop and change the design** (more
  maps, different memory encoding) before M3/M4 cost rendering time.

### M2: Restart action
- `ACTION_DIM` 4, restart head and loss, runtime cooldown, harness statistics.
- Labels from the divergence rule (M3 step 4) on renders of perturbed single attempts
  (no sessions needed yet). Restart = full restart only, no checkpoint respawn.
- Exit: restart precision and recall reach their targets; the false restart rate on good
  runs is low.

### M3: Session data generator
- Plugin `SAVE_STATE` / `LOAD_STATE`, hard-spot detection, naive operator, failure
  detection, `render_sessions.py`, `sessions.jsonl`, quality checks (every attempt is
  deterministic, the failure point is plausible).
- Exit: sessions for at least the training maps, render time measured and documented,
  L1 shows enough memory-dependent ticks.

### M4: Train the memory model (variant B)
- Variant B with memory dropout and hard-spot oversampling (L4).
- Exit: L3 probes on held-out maps: `mem_blank_gap` clearly positive, `mem_swap_follow`
  close to 1, `mem_foreign` like blank memory. If they fail, compare with the M1a FakeGame
  result first (data or model wiring problem vs. TMNF-specific problem).

### M5: Multi-attempt closed-loop eval
- Held-out maps, N attempts per map, several seeds. Improvement curve over the attempt
  index with memory and with the memory blanked (L5).
- Exit: the success criteria at the top are met for at least one variant. Record the
  result in `docs/decisions.md`.

### M6: The agent's own attempts in memory
- Distribution shift: at test time the memory holds the agent's **own** failures, which
  look different from synthetic ones. Generate sessions from agent attempts, relabel
  failure states with replay actions (DAgger style, as in PLAN.md Phase 5) and retrain.

### M7 (optional, only on explicit user request): RL across attempts
- In-context RL or fine-tuning on real learning histories. Outside the scope of PLAN.md
  unless the user asks for it.

## Risks

- The memory is ignored (see above). M1 and the ablation in M5 catch this early.
- Too few maps: the model memorizes maps instead of learning to use the memory (R4).
- Synthetic fails do not look like agent fails. M6 addresses this.
- Rendering cost for sessions. Prefix reuse (M3) and render speed-ups (M0) are required.
- Restart too eager: the agent restarts good attempts and never reaches the finish.
  Needs a class weight, a cooldown and the false restart metric.
- The divergence rule needs a successful source run. At runtime there is none, so the
  model must learn to recognize a lost run from the image alone. M2 checks that.
- Memory frames at 1 fps miss short, decisive spots (a turn-in point). M1 compares 1 fps
  with 0.5 fps; if both are too coarse, test 2 fps with fewer seconds.
- Long contexts: VRAM and latency. The memory is encoded only once per attempt, so the
  60 Hz loop only pays for the attention over cached tokens. Measure the latency budget
  again (`tools/measure_latency.py`).

## Decisions (user, 2026-10-05)

- **Lost run / restart label**: the attempt diverges clearly from the successful source
  run; the attempt ends at the latest 15 s after the input change (M3 step 4).
- **Respawn**: full restart only, no checkpoint respawn as a second action.
- **Training target**: variant B (systematic naive behavior, fixed at spots where an
  earlier attempt crashed), with the learning checks L1-L5 as gates.
- **Memory size**: defined by driving time, not by attempts. At least 3 min, preferably
  5 min: start with 1 fps x 300 frames (5 min); variant 0.5 fps x 300 frames (10 min)
  is compared in M1.

## Open questions

- Hard-spot thresholds and the naive delay range Δ: set in M3 from rendered data (L1
  shows whether the choice yields enough memory-dependent ticks).
- Divergence thresholds (distance, progress lag): set in M2 from rendered data.
