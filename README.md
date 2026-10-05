# Trackmania-Agent

Imitation-learning agent for **TrackMania Nations Forever** (TMNF): a small
vision + action-history transformer that plays live at a 60 Hz control rate,
trained by behavior cloning on replays re-driven in the game and rendered as
low-resolution clips.

Status: code complete and tested on CPU with a fake game; **not yet run on the
real game**. Next step is Phase 0 on a Windows PC with TMNF + TMInterface.

## Documents

| File | Content |
|------|---------|
| `docs/PLAN.md` | requirements and phases (German) |
| `docs/WORKLOG.md` | status board and log, read first when picking up work |
| `docs/ARCHITECTURE.md` | technical spec, contracts, timing, formats |
| `docs/decisions.md` | decision record (what was tried, measured, rejected) |
| `docs/research.md` | verified facts about TMNF tooling with sources |
| `docs/setup_windows.md` | game + TMInterface + plugin setup, Phase 0 checklist |
| `tmagent/game/tmnf/PROTOCOL.md` | plugin <-> Python protocol |

## Pipeline (on the game machine)

```powershell
pip install -e .[dev,game]                     # plus CUDA torch, see docs/setup_windows.md
python -m pytest -q                            # CPU tests, no game needed
python tools/fake_pipeline.py                  # end-to-end proof on the fake game (~1 min)

# Phase 0
python tools/system_report.py                  # -> docs/system.md
python tools/tmnf_smoke.py --config configs/tmnf.yaml --map <map> [--replay <file>]
python tools/measure_latency.py --config configs/tmnf.yaml --map <map>   # -> docs/latency.md

# Phase 1 data (read TMNF EULA + TMX terms first)
python tools/tmx_download.py --i-have-read-the-tmx-terms --user-agent "..." tracks ...
python tools/tmx_download.py --i-have-read-the-tmx-terms --user-agent "..." replays ...
python tools/render_replays.py --config configs/tmnf.yaml --replays data/tmnf/replays
python -m tmagent.data.quality data/tmnf --config configs/tmnf.yaml

# Phase 2/3 training and evaluation
python -m tmagent.train.train_bc --config configs/tmnf.yaml --set data.history_s=0 model.use_action_history=false --set name=baseline
python -m tmagent.train.train_bc --config configs/tmnf.yaml --set name=context_2s
python -m tmagent.eval.harness --config configs/tmnf.yaml --ckpt experiments/<run>/checkpoints/last.pt --set "eval.maps=[...]"
python -m tmagent.runtime.live --config configs/tmnf.yaml --ckpt <ckpt> --map <map> --duration-s 120
```

Ablation configs: `configs/baseline_single_frame.yaml`, `configs/context_2s.yaml`,
`configs/context_2s_no_actions.yaml`; combine with `--set`.

## Layout

```
tmagent/   interfaces, config, data, model, train, runtime, eval, game (fake + tmnf)
tools/     render_replays, fake_pipeline, system_report, measure_latency, tmx_download, tmnf_smoke
configs/   YAML configs
tests/     pytest (CPU only)
```

Rules: never submit runs made with TMInterface to online leaderboards; check
the TMNF EULA and TMX terms before downloading or automating (docs/research.md).
