# System report (Phase 0.1)

Generated 2026-10-05T17:54:09 by `tools/system_report.py`.

## Machine

| item | value |
|---|---|
| OS | Windows-10-10.0.19045-SP0 |
| CPU | Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz (8 logical cores, AMD64) |
| RAM | 23.9 GiB |
| Python | 3.13.2 (CPython) |
| torch | 2.6.0+cu124 |
| torch CUDA | available (build 12.4, cudnn 90100) |
| bf16 on GPU | yes |
| torch CPU threads | 4 |

## GPU

| # | name | VRAM GiB | compute capability | SMs |
|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 4060 | 8.0 | 8.9 | 24 |

`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`:

```
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4060, 8188 MiB, 610.88
```

## Capacity estimate (ESTIMATE, to be replaced by measurement)

Basis: GPU `NVIDIA GeForce RTX 4060` (8.0 GiB VRAM). Formula-based only; real limits depend on the model, sequence length, batch size and framework overhead. Measure with `tools/measure_latency.py` and the `[model]`/`[speed]` lines of `tmagent.train.train_bc`.

- bf16 inference: ~2 B/param + 20% overhead + 1 GiB reserve -> max **3.13 B** parameters
- full AdamW training: ~16 B/param (fp32 weights, grads, two moments) with 50% of the memory kept free for activations -> max **268 M** parameters

| parameters | bf16 inference VRAM GiB | AdamW training VRAM GiB (incl. headroom) | inference fits | training fits |
|---|---|---|---|---|
| 50 M | 1.1 | 1.5 | yes | yes |
| 300 M | 1.7 | 8.9 | yes | no |
| 500 M | 2.1 | 14.9 | yes | no |
| 1.00 B | 3.2 | 29.8 | yes | no |
| 3.00 B | 7.7 | 89.4 | yes | no |
