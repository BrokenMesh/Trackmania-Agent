# Latency report (Phase 0.4 / 0.5)

Generated 2026-10-05T18:40:19 by `tools/measure_latency.py`. Config `tmnf`, game backend `tmnf`, device `cuda` (NVIDIA GeForce RTX 4060), precision `bf16`, 300 timed iterations per measurement (30 warm-up).

Machine: Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz, torch 2.6.0+cu124.

## Game

| section | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| grab_frame | 300 | 13.317 | 17.014 | 18.015 | 19.824 |
| rt_frame_age | 176 | 0.619 | 1.064 | 1.342 | 16.652 |
| rt_frame_interval | 175 | 28.738 | 33.418 | 34.740 | 35.078 |
| rt_set_action | 300 | 0.002 | 0.002 | 0.004 | 0.079 |
| step_1_tick | 300 | 7.521 | 10.400 | 12.190 | 12.992 |

Map `A01-Race`. Sync step/grab are what `render_replays` pays per tick/frame: about 1.02 s of wall time per second of race time (100 steps + frame_hz grabs + preprocess, p50).

## Preprocess (conform_frame + tensor conversion)

| section | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| game_frame/conform_frame | 300 | 0.004 | 0.007 | 0.008 | 0.015 |
| game_frame/to_tensor | 300 | 0.200 | 0.474 | 0.567 | 254.048 |
| game_frame/total | 300 | 0.204 | 0.482 | 0.572 | 254.058 |
| window_size_frame/conform_frame | 300 | 7.200 | 30.113 | 39.188 | 72.999 |
| window_size_frame/to_tensor | 300 | 0.506 | 0.790 | 0.851 | 0.889 |
| window_size_frame/total | 300 | 7.817 | 30.527 | 39.911 | 73.599 |

## Model chain (random init, per iteration)

| model | params | K | tokens/frame | capture p99 | preprocess p99 | observe p99 | predict p99 | chain p50 | chain p95 | chain p99 | peak mem MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 0.45 M | 11 | 4 | 12.51 | 0.94 | 3.58 | 4.89 | 15.94 | 18.52 | 20.68 | 12 |
| baseline_single_frame | 3.61 M | 1 | 16 | 11.95 | 1.15 | 4.76 | 7.40 | 17.77 | 20.83 | 22.46 | 29 |
| context_2s | 3.68 M | 41 | 16 | 10.23 | 0.96 | 4.37 | 10.66 | 19.24 | 22.50 | 23.60 | 45 |
| wide_deep | 25.91 M | 41 | 16 | 10.45 | 0.86 | 3.50 | 12.10 | 19.36 | 22.83 | 23.81 | 197 |

## Budget (control 60 Hz, frames 20 Hz)

- inference period = 1 / frame_hz, chunk horizon = chunk_len / control_hz, latency = chain p99 (capture + preprocess + observe + predict).
- fits = latency <= period and margin (horizon - latency) >= 50% of the horizon.
- min chunk_len = smallest chunk with that margin (latency <= 50% of the horizon); gapless chunk_len = also covers one inference period, so the next chunk arrives before the current one ends. Both are at least control_hz / frame_hz.

| model | params | latency p99 ms | period ms | horizon ms | margin ms (%) | fits period | margin ok | fits | min chunk_len | gapless chunk_len |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 0.45 M | 20.68 | 50.0 | 133.3 | 112.7 (84 %) | yes | yes | **yes** | 3 | 5 |
| baseline_single_frame | 3.61 M | 22.46 | 50.0 | 133.3 | 110.9 (83 %) | yes | yes | **yes** | 3 | 5 |
| context_2s | 3.68 M | 23.60 | 50.0 | 133.3 | 109.7 (82 %) | yes | yes | **yes** | 3 | 5 |
| wide_deep | 25.91 M | 23.81 | 50.0 | 133.3 | 109.5 (82 %) | yes | yes | **yes** | 3 | 5 |

Recommendation: largest fitting model = **wide_deep**.

## All profiler sections

| section | n | mean ms | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|---|
| game/grab_frame | 300 | 13.518 | 13.317 | 17.014 | 18.015 | 19.824 |
| game/rt_frame_age | 176 | 0.715 | 0.619 | 1.064 | 1.342 | 16.652 |
| game/rt_frame_interval | 175 | 28.590 | 28.738 | 33.418 | 34.740 | 35.078 |
| game/rt_set_action | 300 | 0.002 | 0.002 | 0.002 | 0.004 | 0.079 |
| game/step_1_tick | 300 | 7.686 | 7.521 | 10.400 | 12.190 | 12.992 |
| model/baseline_single_frame/capture | 300 | 9.556 | 9.464 | 10.854 | 11.954 | 25.965 |
| model/baseline_single_frame/chain | 300 | 18.115 | 17.775 | 20.833 | 22.460 | 37.734 |
| model/baseline_single_frame/observe | 300 | 2.559 | 2.334 | 3.996 | 4.758 | 5.331 |
| model/baseline_single_frame/predict | 300 | 5.261 | 5.174 | 6.413 | 7.403 | 12.914 |
| model/baseline_single_frame/preprocess | 300 | 0.740 | 0.715 | 0.907 | 1.145 | 1.233 |
| model/context_2s/capture | 300 | 8.382 | 8.414 | 9.816 | 10.227 | 11.969 |
| model/context_2s/chain | 300 | 19.474 | 19.245 | 22.498 | 23.604 | 25.353 |
| model/context_2s/observe | 300 | 2.482 | 2.257 | 3.696 | 4.366 | 5.018 |
| model/context_2s/predict | 300 | 7.938 | 7.731 | 10.018 | 10.659 | 12.497 |
| model/context_2s/preprocess | 300 | 0.672 | 0.652 | 0.787 | 0.956 | 1.147 |
| model/smoke/capture | 300 | 9.955 | 9.703 | 11.302 | 12.514 | 139.035 |
| model/smoke/chain | 300 | 16.339 | 15.943 | 18.518 | 20.677 | 145.743 |
| model/smoke/observe | 300 | 2.362 | 2.246 | 3.137 | 3.576 | 4.167 |
| model/smoke/predict | 300 | 3.424 | 3.323 | 4.361 | 4.890 | 5.481 |
| model/smoke/preprocess | 300 | 0.597 | 0.587 | 0.743 | 0.941 | 1.083 |
| model/wide_deep/capture | 300 | 7.444 | 7.317 | 9.359 | 10.455 | 10.833 |
| model/wide_deep/chain | 300 | 19.496 | 19.357 | 22.826 | 23.811 | 26.471 |
| model/wide_deep/observe | 300 | 2.347 | 2.232 | 3.013 | 3.503 | 3.816 |
| model/wide_deep/predict | 300 | 9.142 | 9.027 | 11.642 | 12.102 | 12.869 |
| model/wide_deep/preprocess | 300 | 0.563 | 0.533 | 0.741 | 0.858 | 1.022 |
| preprocess/game_frame/conform_frame | 300 | 0.004 | 0.004 | 0.007 | 0.008 | 0.015 |
| preprocess/game_frame/to_tensor | 300 | 1.090 | 0.200 | 0.474 | 0.567 | 254.048 |
| preprocess/game_frame/total | 300 | 1.094 | 0.204 | 0.482 | 0.572 | 254.058 |
| preprocess/window_size_frame/conform_frame | 300 | 10.743 | 7.200 | 30.113 | 39.188 | 72.999 |
| preprocess/window_size_frame/to_tensor | 300 | 0.541 | 0.506 | 0.790 | 0.851 | 0.889 |
| preprocess/window_size_frame/total | 300 | 11.284 | 7.817 | 30.527 | 39.911 | 73.599 |
