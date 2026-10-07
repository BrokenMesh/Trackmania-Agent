# Phase 1 learnings (input for Phase 2)

Everything here was measured on the user's machine (TMUF 2.12, TMLoader, TMInterface 2.2.1,
i7-4790 4C/8T, RTX 4060 8 GB, 24 GB RAM, episodes on an HDD). Status markers: VERIFIED =
measured / reproduced, LIKELY = consistent evidence but not isolated. Decisions with details:
`docs/decisions.md` (D-014 .. D-017). Chronology: `docs/WORKLOG.md`.

## 1. Game bridge and simulation

- VERIFIED: TMI sync stepping is deterministic. A keyboard replay re-driven tick by tick
  reproduces the finish time exactly (+-0 ms) and the images the eval loop captures equal
  the training frames (mean abs pixel difference 0.7 / 255, render noise only).
- VERIFIED quirks (all handled in code): replay inputs lead by 10 ms; left+right held ->
  left wins; the finish is reported one step late; the saved t=0 state can be bad right
  after a map load (reload fixes it); maps copied after game start are unknown until a
  restart; the map intro needs window focus; some community intros wait for Enter; one
  community map hangs the loader forever (black screen) -> `bad_map` skip list.
- VERIFIED: TMInterface loads plugins at game start only; a plugin change needs a game
  restart. Game start sequence: TMLoader Play -> skip DirectPlay popup -> Stay offline ->
  profile `elkor` (no records, so no author ghost in the frames).
- VERIFIED: analog (gamepad) replays do not re-drive deterministically through
  SetInputState; keyboard-only replays do. Respawn runs desync as well (excluded).
- Several game instances work (plugin port fallback 8477..8484, `--shard I/N`), but the
  4-core CPU caps the total: 2 instances ~5x real time, 3 instances the same (5.07x).
  Rendering plus GPU training at the same time makes the PC stutter.

## 2. Data

- Dataset after Phase 1: 3,585 human episodes, 51.6 h, 259 maps (64 campaign + 197
  top-awarded TMX community maps, author time 15-60 s), split by map 214 / 19 / 26.
- 78 GB on disk at 128x96 RGB-ish frames, 20 Hz (1.5 GB per hour of driving). It does not
  fit in RAM; random window reads from the HDD cost ~0.15 s/step (0.55 vs 0.43 s/step).
  Phase 2 with 300-frame memories will be I/O bound on this disk: plan an SSD or a
  lower-resolution memory stream.
- Keyboard labels are 60 % "no steer", 3 % brake. Regression on them collapses to 0
  (D-016); a 3-class steering head with PWM execution is required.

## 3. Behaviour cloning results

| Model | data | closed loop: 26 unseen test maps (median / mean progress) | finishes |
|---|---|---|---|
| single frame, discrete3 | 7.8 h | 0.047 / 0.058 | 0 |
| 2 s context, step 20k | 7.8 h | 0.046 / 0.058 | 0 |
| 2 s context, step 40k | 51.6 h | 0.051 / 0.067 | 0 (A03, a train map, finished once) |

- VERIFIED: 6.6x more human data did not move closed-loop progress beyond noise, although
  offline the model is good on unseen maps (steer sign accuracy 0.88, corr 0.63, gas 0.96).
- VERIFIED: the closed-loop gap is not a pipeline bug (section 1, frames identical; live
  vs offline predictions on the same frames corr 0.992). It is compounding error: after a
  small deviation the car is in states no human demonstrated, and there are no recovery
  demonstrations. Typical failure: turn-in slightly late -> wall -> stuck with full gas.
- VERIFIED: val loss bottoms early (step 9-12k of 40k) while train loss keeps falling;
  val loss is not predictive of closed-loop driving. Judge models in closed loop only.
- Eval noise: one attempt per map, 26-30 maps; differences below ~0.02 mean progress are
  noise. The progress measure jumps on stunt maps whose path overlaps itself (A04, A10):
  exclude acrobatic maps from progress averages or use checkpoint counts.

## 4. Recovery data (DAgger with a line expert)

- `tmagent/eval/expert.py` LineExpert: follows the fastest human run of the map; copies the
  human input while within 1.5 m and aligned, otherwise pure pursuit (binary steer with a
  dead band) towards a point 0.45 s ahead. Alone it finishes 4 of A01-A10 (agent: 1) at
  ~1.5x the human time; average progress 0.79. Lookahead 0.3 / 0.7 s and a wider
  "on line" band were worse.
- `tmagent/eval/dagger.py`: mixed control, labels always from the expert.
  VERIFIED: expert-takes-over-at-4 m alone does not work (the agent is already in a wall,
  runs end at 1-7 % progress). Alternating random blocks (agent 0.5-2 s, expert 1-2 s) plus
  intervention at 2.5 m, and ending a run after 4 s without progress, gives usable runs.
- VERIFIED (3 attempts per map, steer noise 0.05): one DAgger round (1.24 h, 23 % of the
  training samples) lifts A01-A10 median progress 0.44 -> 0.60 (A01 0.05 -> 0.95, A08
  0.03 -> 0.38). Two more rounds with a growing expert share (36-45 %) fell back to 0.43:
  the expert drives worse than the humans (bang-bang steering, ~1.5x the human time), and
  too much of it pulls the policy towards it. Keep the expert share near 20 %. All three
  rounds at 23 % gave 0.45: more of this expert's data does not help either.
- VERIFIED: no round transferred to unseen maps (test mean progress 0.07-0.08 throughout).
  Recoveries learned on a map stay on that map. Transfer needs recovery data on far more
  maps and/or better recovery labels (human takeovers).
- VERIFIED: the simulation is deterministic, so one attempt per map is a point sample that
  flips with tiny weight changes. Evaluate with >= 3 attempts and small action noise.

## 5. Implications for Phase 2 (memory agent)

- The memory agent is built on top of a driving policy that currently cannot recover from
  its own mistakes. Phase 2 M1 (prove the memory signal) should start from a policy that
  survives typical errors, otherwise every "attempt" in memory ends in the same first-corner
  crash and there is little to learn across attempts.
- The machinery for synthetic failed attempts exists now: the LineExpert gives a
  corrective label anywhere near the line, the DAgger collector produces agent-style
  deviations, and takeover mode (`tools/eval_ui.py`, plugin passthrough) records human
  recoveries. PLAN_PHASE2 M3 (session generator) can reuse these instead of a new expert.
- Phase 2 synthetic failed attempts: a DAgger-style run (agent blocks, then the expert)
  produces realistic "went wrong here" moments for free; the memory model can learn from
  pairs of a failed and a corrected pass over the same section.
- "Lost run" detection for Phase 2 can use the expert distance to the reference line
  (> 2.5 m, or no progress for 4 s) - the same signals that work for DAgger.
- Budget realities: 1 game instance runs sync stepping at ~1x real time with a policy in
  the loop; 2 instances are the useful maximum; 40k training steps take ~6.5 h.
