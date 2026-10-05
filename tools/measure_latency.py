"""Phase 0.4 / 0.5: latency of the chain capture -> preprocess -> model -> input, plus budget.

    python tools/measure_latency.py --config configs/tmnf.yaml --map <uid or file> \
        [--models smoke context_2s "mine:d_model=512,n_layers=8"] [--quick] [--out docs/latency.md]

Measures with tmagent.runtime.profiler.LatencyProfiler (p50 / p95 / p99):
- game (cfg.game.backend): sync step + grab_frame, then realtime set_action, frame interval
  and frame age;
- preprocess: tmagent.data.render.conform_frame + tensor conversion;
- model: StreamingPolicy observe + predict of a random-init TMPolicy for each model spec,
  chain = capture + preprocess + observe + predict measured per iteration;
- budget: chunk horizon (chunk_len / control_hz) and inference period (1 / frame_hz) against
  the chain p99, the largest model that fits and the minimum chunk_len of each model.

A model spec is `name` (overlay from configs/<name>.yaml: its model section and
data.history_s / data.chunk_len) or `name:key=value,...` (`key` = model key, or `data.key`;
`@file.yaml` inside the list loads such an overlay). The default sweep is smoke,
baseline_single_frame, context_2s (the configs/*.yaml overlays) and a wider/deeper variant.
Writes the markdown report (default docs/latency.md) and the same data as JSON next to it.
Not measured here: the model running concurrently with the 60 Hz control thread (see
tmagent.runtime.live / tools/fake_pipeline.py for deadline misses under load).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from tmagent.config import Config, load_config
from tmagent.data.render import conform_frame
from tmagent.interfaces import Action
from tmagent.runtime.profiler import LatencyProfiler
from tools import system_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "latency.md"
DEFAULT_MODELS = (
    "smoke",
    "baseline_single_frame",
    "context_2s",
    "wide_deep:d_model=512,n_layers=8,n_heads=8,data.history_s=2.0",
)
OVERLAY_DATA_KEYS = ("history_s", "chunk_len")  # data keys taken from a config overlay
MARGIN_FRAC = 0.5  # the chain must leave this fraction of the chunk horizon as margin


# ---------------------------------------------------------------- model specs


@dataclasses.dataclass
class ModelSpec:
    name: str
    model: dict[str, Any] = dataclasses.field(default_factory=dict)
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


def _overlay(path: Path, spec: ModelSpec) -> None:
    raw = yaml.safe_load(path.read_text()) or {}
    spec.model.update(raw.get("model") or {})
    spec.data.update({k: v for k, v in (raw.get("data") or {}).items() if k in OVERLAY_DATA_KEYS})


def parse_model_spec(text: str, root: Path = ROOT) -> ModelSpec:
    """`name`, `name:key=val,key=val` or `name:@overlay.yaml,...` -> ModelSpec."""
    name, colon, rest = text.partition(":")
    spec = ModelSpec(name.strip())
    if not colon:
        path = root / "configs" / f"{spec.name}.yaml"
        if not path.is_file():
            raise ValueError(f"model spec {text!r}: no {path} (use name:key=value,...)")
        _overlay(path, spec)
        return spec
    for item in (x.strip() for x in rest.split(",")):
        if not item:
            continue
        if item.startswith("@"):
            path = Path(item[1:])
            _overlay(path if path.is_absolute() else root / path, spec)
            continue
        key, eq, val = item.partition("=")
        if not eq:
            raise ValueError(f"model spec {text!r}: {item!r} is not key=value")
        section, _, k = key.strip().rpartition(".")
        section = section or "model"
        if section not in ("model", "data"):
            raise ValueError(f"model spec {text!r}: unknown section {section!r}")
        getattr(spec, section)[k] = yaml.safe_load(val)
    return spec


def apply_spec(cfg: Config, spec: ModelSpec) -> Config:
    """Copy of cfg with the spec's model/data overrides (validated)."""
    out = copy.deepcopy(cfg)
    try:
        out.model = dataclasses.replace(out.model, **spec.model)
        out.data = dataclasses.replace(out.data, **spec.data)
    except TypeError as exc:
        raise ValueError(f"model spec {spec.name!r}: {exc}") from exc
    out.validate()
    return out


# ---------------------------------------------------------------- measuring helpers


def timed(
    prof: LatencyProfiler, name: str, fn: Callable[..., Any], *args: Any
) -> tuple[Any, float]:
    """Call fn(*args); record the elapsed time under `name`; returns (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args)
    dt = time.perf_counter() - t0
    prof.record(name, dt)
    return out, dt


def _sync(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def to_tensor(img: np.ndarray, device: Any) -> Any:
    """uint8 (H, W, C) -> float [1, C, H, W] in [0, 1] on `device`."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    x = x.permute(2, 0, 1)[None].float().div_(255.0)
    _sync(device)
    return x


def synthetic_frame(h: int, w: int, c: int = 3) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, c), dtype=np.uint8)


def resolve_device(name: str) -> Any:
    """Device from cfg.runtime.device; falls back to CPU (with a warning) without CUDA."""
    import torch

    from tmagent.model.policy import resolve_device as resolve

    dev = resolve(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, measuring on CPU")
        return torch.device("cpu")
    return dev


def default_map(cfg: Config, given: str | None) -> str | None:
    if given:
        return given
    if cfg.game.backend == "fake":
        t = cfg.game.fake_track
        return t if t.startswith("fake:") else f"fake:{t}"
    return cfg.eval.maps[0] if cfg.eval.maps else None


# ---------------------------------------------------------------- game


def measure_game_sync(
    game: Any, cfg: Config, map_ref: str, prof: LatencyProfiler, n: int
) -> np.ndarray:
    """Sync step(1 tick) and grab_frame latency over n iterations; returns a sample frame."""
    game.load_map(map_ref)
    game.start_race()
    drive = Action(0.0, 1.0, 0.0)
    frame = None
    for _ in range(n):
        timed(prof, "game/step_1_tick", game.step, drive, 1)
        frame, _ = timed(prof, "game/grab_frame", game.grab_frame)
    assert frame is not None
    return frame.image


def measure_game_realtime(
    game: Any, map_ref: str, prof: LatencyProfiler, n_actions: int, seconds: float
) -> dict[str, Any]:
    """Realtime set_action latency, frame interval and frame age.

    Frames are polled every ~0.5 ms (coarser on Windows with Python < 3.11, which inflates
    the measured age by up to the sleep granularity).
    """
    game.load_map(map_ref)
    game.restart()
    try:
        time.sleep(0.2)  # let the capture thread produce frames
        for _ in range(n_actions):
            timed(prof, "game/rt_set_action", game.set_action, Action(0.1, 1.0, 0.0))
        last_wall: float | None = None
        deadline = time.perf_counter() + seconds
        frames = 0
        while time.perf_counter() < deadline:
            frame = game.latest_frame()
            if frame is not None and frame.wall_time != last_wall:
                now = time.perf_counter()
                prof.record("game/rt_frame_age", max(now - frame.wall_time, 0.0))
                if last_wall is not None:
                    prof.record("game/rt_frame_interval", frame.wall_time - last_wall)
                last_wall = frame.wall_time
                frames += 1
            time.sleep(0.0005)
    finally:
        game.set_action(Action())  # leave the game neutral; the caller closes or reloads it
    return {"frames": frames, "seconds": seconds}


# ---------------------------------------------------------------- preprocess


def measure_preprocess(
    cfg: Config, frame: np.ndarray, device: Any, prof: LatencyProfiler, n: int
) -> None:
    """conform_frame + tensor conversion of a captured frame and of a window-size frame."""
    win_w, win_h = cfg.game.window_size
    sources = {"game_frame": frame}
    if frame.shape[:2] != (win_h, win_w):
        sources["window_size_frame"] = synthetic_frame(win_h, win_w, 3)
    for label, src in sources.items():
        for _ in range(n):
            t0 = time.perf_counter()
            img, _ = conform_frame(src, cfg.data)
            t1 = time.perf_counter()
            to_tensor(img, device)
            t2 = time.perf_counter()
            prof.record(f"preprocess/{label}/conform_frame", t1 - t0)
            prof.record(f"preprocess/{label}/to_tensor", t2 - t1)
            prof.record(f"preprocess/{label}/total", t2 - t0)


# ---------------------------------------------------------------- model chain


def measure_model(
    spec: ModelSpec,
    cfg: Config,
    game: Any | None,
    device: Any,
    prof: LatencyProfiler,
    n: int,
    warmup: int,
) -> dict[str, Any]:
    """Random-init model: capture -> preprocess -> observe -> predict, n timed iterations."""
    import torch

    from tmagent.model.policy import TMPolicy, count_parameters
    from tmagent.model.streaming import StreamingPolicy

    mcfg = apply_spec(cfg, spec)
    d = mcfg.data
    model = TMPolicy(mcfg.model, d)
    counts = count_parameters(model)
    policy = StreamingPolicy(model, d, device, cfg.runtime.precision)
    rng = np.random.default_rng(0)
    past = rng.uniform(-1, 1, (d.actions_per_frame, 3)).astype(np.float32)
    fallback = synthetic_frame(cfg.game.window_size[1], cfg.game.window_size[0], 3)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pre = f"model/{spec.name}"

    def capture() -> np.ndarray:
        return game.grab_frame().image if game is not None else fallback

    def step(record: bool) -> None:
        t0 = time.perf_counter()
        image = capture()
        t1 = time.perf_counter()
        img, _ = conform_frame(image, d)
        to_tensor(img, device)
        t2 = time.perf_counter()
        policy.observe(img, past)
        _sync(device)
        t3 = time.perf_counter()
        policy.predict()
        t4 = time.perf_counter()
        if record:
            for key, dt in (
                ("capture", t1 - t0),
                ("preprocess", t2 - t1),
                ("observe", t3 - t2),
                ("predict", t4 - t3),
                ("chain", t4 - t0),
            ):
                prof.record(f"{pre}/{key}", dt)

    policy.reset()
    for _ in range(d.num_steps):  # fill the history buffer (observe only, cheap)
        policy.observe(conform_frame(capture(), d)[0], past)
    for _ in range(warmup):
        step(False)
    for _ in range(n):
        step(True)
    row: dict[str, Any] = {
        "name": spec.name,
        "params": counts["total"],
        "encoder_params": counts["encoder"],
        "k_steps": d.num_steps,
        "tokens_per_frame": mcfg.model.tokens_per_frame,
        "d_model": mcfg.model.d_model,
        "n_layers": mcfg.model.n_layers,
        "chunk_len": d.chunk_len,
        "history_s": d.history_s,
        "use_action_history": mcfg.model.use_action_history,
        "peak_mem_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    }
    for key in ("capture", "preprocess", "observe", "predict", "chain"):
        row[key] = prof.stat(f"{pre}/{key}")
    return row


# ---------------------------------------------------------------- budget


def budget(cfg: Config, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Chunk horizon / inference period budget per model and the recommendation.

    horizon = chunk_len / control_hz, period = 1 / frame_hz, latency = chain p99. A model
    fits if latency <= period (inference keeps up with the frame rate) and the margin
    horizon - latency is at least MARGIN_FRAC of the horizon. min_chunk_len is the smallest
    chunk with that margin (horizon >= latency / (1 - MARGIN_FRAC)); gapless_chunk_len also
    covers one inference period after the latency, so the next chunk arrives before the
    current one runs out. Both are at least control_hz / frame_hz (one frame step).
    """
    hz, period_ms = cfg.data.control_hz, 1000.0 / cfg.data.frame_hz
    per_frame = cfg.data.actions_per_frame  # a chunk must cover at least one frame step
    out_rows = []
    for r in rows:
        lat = r["chain"]["p99_ms"]
        horizon = r["chunk_len"] / hz * 1000.0
        margin = horizon - lat
        fits_period = lat <= period_ms
        margin_ok = margin >= MARGIN_FRAC * horizon
        out_rows.append(
            {
                "name": r["name"],
                "params": r["params"],
                "latency_p99_ms": lat,
                "period_ms": period_ms,
                "horizon_ms": horizon,
                "margin_ms": margin,
                "margin_pct": 100.0 * margin / horizon if horizon > 0 else 0.0,
                "fits_period": fits_period,
                "margin_ok": margin_ok,
                "fits": fits_period and margin_ok,
                "min_chunk_len": max(per_frame, math.ceil(lat / 1000.0 * hz / (1 - MARGIN_FRAC))),
                "gapless_chunk_len": max(per_frame, math.ceil((lat + period_ms) / 1000.0 * hz)),
            }
        )
    fitting = [r for r in out_rows if r["fits"]]
    best = max(fitting, key=lambda r: r["params"]) if fitting else None
    return {
        "control_hz": hz,
        "frame_hz": cfg.data.frame_hz,
        "margin_frac": MARGIN_FRAC,
        "rows": out_rows,
        "recommended": best["name"] if best else None,
    }


# ---------------------------------------------------------------- report


def _stat_row(label: str, st: dict[str, float] | None) -> str:
    if st is None:
        return f"| {label} | - | - | - | - | - |"
    return (
        f"| {label} | {st['n']} | {st['p50_ms']:.3f} | {st['p95_ms']:.3f} | "
        f"{st['p99_ms']:.3f} | {st['max_ms']:.3f} |"
    )


def _prefix_table(prof: LatencyProfiler, prefix: str) -> list[str]:
    head = ["| section | n | p50 ms | p95 ms | p99 ms | max ms |", "|---|---|---|---|---|---|"]
    sections = prof.summary()["sections"]
    rows = [_stat_row(k[len(prefix) :], v) for k, v in sections.items() if k.startswith(prefix)]
    return head + rows if rows else ["No samples."]


def render_markdown(res: dict[str, Any], prof: LatencyProfiler) -> str:
    m, b = res["meta"], res["budget"]
    lines = [
        "# Latency report (Phase 0.4 / 0.5)",
        "",
        f"Generated {m['date']} by `tools/measure_latency.py`. Config `{m['config']}`, game "
        f"backend `{m['backend']}`, device `{m['device']}` ({m['device_name']}), precision "
        f"`{m['precision']}`, {m['n']} timed iterations per measurement ({m['warmup']} warm-up).",
        "",
        f"Machine: {m['cpu']}, torch {m['torch']}.",
    ]
    if m["backend"] == "fake" or m["device"] == "cpu":
        lines += [
            "",
            "> NOTE: fake game and/or CPU measurement. This proves the tool works; the budget "
            "below is only meaningful when measured with the real game on the target GPU.",
        ]
    lines += ["", "## Game", ""]
    if res["game"].get("error"):
        lines.append(f"Game part skipped: {res['game']['error']}")
    else:
        lines += _prefix_table(prof, "game/")
        lines += [
            "",
            f"Map `{res['game']['map']}`. Sync step/grab are what `render_replays` pays per "
            f"tick/frame: about {res['game']['render_s_per_race_s']:.3g} s of wall time per "
            "second of race time (100 steps + frame_hz grabs + preprocess, p50).",
        ]
    lines += ["", "## Preprocess (conform_frame + tensor conversion)", ""]
    lines += _prefix_table(prof, "preprocess/")
    lines += ["", "## Model chain (random init, per iteration)", ""]
    lines += [
        "| model | params | K | tokens/frame | capture p99 | preprocess p99 | observe p99 | "
        "predict p99 | chain p50 | chain p95 | chain p99 | peak mem MiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["models"]:
        mem = f"{r['peak_mem_mib']:.0f}" if r["peak_mem_mib"] is not None else "-"
        lines.append(
            f"| {r['name']} | {r['params'] / 1e6:.2f} M | {r['k_steps']} | "
            f"{r['tokens_per_frame']} | {r['capture']['p99_ms']:.2f} | "
            f"{r['preprocess']['p99_ms']:.2f} | {r['observe']['p99_ms']:.2f} | "
            f"{r['predict']['p99_ms']:.2f} | {r['chain']['p50_ms']:.2f} | "
            f"{r['chain']['p95_ms']:.2f} | {r['chain']['p99_ms']:.2f} | {mem} |"
        )
    lines += [
        "",
        f"## Budget (control {b['control_hz']} Hz, frames {b['frame_hz']} Hz)",
        "",
        "- inference period = 1 / frame_hz, chunk horizon = chunk_len / control_hz, latency = "
        "chain p99 (capture + preprocess + observe + predict).",
        f"- fits = latency <= period and margin (horizon - latency) >= {b['margin_frac']:.0%} "
        "of the horizon.",
        f"- min chunk_len = smallest chunk with that margin (latency <= "
        f"{1 - b['margin_frac']:.0%} of the horizon); gapless chunk_len = also covers one "
        "inference period, so the next chunk arrives before the current one ends. Both are "
        "at least control_hz / frame_hz.",
        "",
        "| model | params | latency p99 ms | period ms | horizon ms | margin ms (%) | fits "
        "period | margin ok | fits | min chunk_len | gapless chunk_len |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in b["rows"]:
        lines.append(
            f"| {r['name']} | {r['params'] / 1e6:.2f} M | {r['latency_p99_ms']:.2f} | "
            f"{r['period_ms']:.1f} | {r['horizon_ms']:.1f} | {r['margin_ms']:.1f} "
            f"({r['margin_pct']:.0f} %) | {_yn(r['fits_period'])} | {_yn(r['margin_ok'])} | "
            f"**{_yn(r['fits'])}** | {r['min_chunk_len']} | {r['gapless_chunk_len']} |"
        )
    rec = b["recommended"]
    lines += [
        "",
        f"Recommendation: largest fitting model = **{rec}**."
        if rec
        else "Recommendation: no model fits the budget; shrink the model, lower frame_hz or "
        "raise chunk_len (see the min chunk_len column).",
        "",
        "## All profiler sections",
        "",
        prof.to_markdown(),
    ]
    return "\n".join(lines)


def _yn(ok: bool) -> str:
    return "yes" if ok else "no"


# ---------------------------------------------------------------- main flow


def run(
    cfg: Config,
    specs: list[ModelSpec],
    map_ref: str | None,
    n: int,
    warmup: int,
    n_grabs: int,
    rt_seconds: float,
    use_game: bool = True,
    game: Any | None = None,
) -> tuple[dict[str, Any], LatencyProfiler]:
    """All measurements; returns (result dict, profiler). `game` overrides make_game (tests)."""
    import torch

    prof = LatencyProfiler()
    device = resolve_device(cfg.runtime.device)
    info = system_report.torch_info()
    res: dict[str, Any] = {
        "meta": {
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "config": cfg.name,
            "backend": cfg.game.backend,
            "device": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "precision": cfg.runtime.precision if device.type == "cuda" else "fp32",
            "n": n,
            "warmup": warmup,
            "cpu": system_report.cpu_name(),
            "torch": info.get("version", "?"),
        },
        "game": {},
    }
    own_game = False
    sample: np.ndarray | None = None
    if use_game and map_ref is None:
        res["game"]["error"] = "no map given (--map, or eval.maps in the config)"
        use_game = False
    if use_game and game is None:
        from tmagent.game import make_game

        try:
            game = make_game(cfg.game, cfg.data)
            own_game = True
        except Exception as exc:
            res["game"]["error"] = f"could not start the game: {type(exc).__name__}: {exc}"
            use_game = False
    if not use_game:
        game = None
        res["game"].setdefault("error", "disabled (--no-game): synthetic frames, capture = 0")
    try:
        if game is not None and map_ref is not None:
            print(f"game: sync step/grab x{n_grabs} on {map_ref}")
            sample = measure_game_sync(game, cfg, map_ref, prof, n_grabs)
            res["game"]["map"] = map_ref
            print(f"game: realtime set_action / frames for {rt_seconds:.1f} s")
            try:
                res["game"]["realtime"] = measure_game_realtime(
                    game, map_ref, prof, max(n_grabs, 200), rt_seconds
                )
            except Exception as exc:
                res["game"]["realtime_error"] = f"{type(exc).__name__}: {exc}"
                print(f"realtime part failed: {res['game']['realtime_error']}")
            game.load_map(map_ref)  # back to sync mode for the model chain
            game.start_race()
        print("preprocess")
        frame = (
            sample
            if sample is not None
            else synthetic_frame(cfg.game.window_size[1], cfg.game.window_size[0])
        )
        measure_preprocess(cfg, frame, device, prof, n)
        rows = []
        for spec in specs:
            print(f"model {spec.name} ({n} iterations)")
            rows.append(measure_model(spec, cfg, game, device, prof, n, warmup))
            print(
                f"  {rows[-1]['params'] / 1e6:.2f} M params, chain p99 "
                f"{rows[-1]['chain']['p99_ms']:.2f} ms"
            )
    finally:
        if own_game and game is not None:
            game.close()
    if "map" in res["game"]:
        g = prof.stat
        step, grab = g("game/step_1_tick"), g("game/grab_frame")
        pre = g("preprocess/game_frame/total")
        if step and grab and pre:
            res["game"]["render_s_per_race_s"] = (
                100 * step["p50_ms"] + cfg.data.frame_hz * (grab["p50_ms"] + pre["p50_ms"])
            ) / 1000.0
    res["game"].setdefault("render_s_per_race_s", float("nan"))
    res["models"] = rows
    res["budget"] = budget(cfg, rows)
    res["profiler"] = prof.summary()
    return res, prof


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="k=v"
    )
    ap.add_argument("--map", default=None, help="map ref (tmnf: uid or file; fake default: oval)")
    ap.add_argument("--models", nargs="+", default=None, help="model specs (see --help text)")
    ap.add_argument("--n", type=int, default=300, help="timed iterations per measurement")
    ap.add_argument("--warmup", type=int, default=30, help="warm-up iterations per model")
    ap.add_argument("--rt-seconds", type=float, default=5.0, help="realtime frame polling time")
    ap.add_argument("--quick", action="store_true", help="small N, short realtime polling")
    ap.add_argument("--no-game", action="store_true", help="skip the game, synthetic frames")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"markdown output ({DEFAULT_OUT})")
    args = ap.parse_args(argv)

    n, warmup, rt = (20, 3, 1.0) if args.quick else (args.n, args.warmup, args.rt_seconds)
    cfg = load_config(args.config, args.overrides)
    specs = [parse_model_spec(s) for s in (args.models or DEFAULT_MODELS)]
    for spec in specs:
        apply_spec(cfg, spec)  # validate before measuring anything
    res, prof = run(
        cfg,
        specs,
        default_map(cfg, args.map),
        n,
        warmup,
        n_grabs=min(n, 30) if args.quick else n,
        rt_seconds=rt,
        use_game=not args.no_game,
    )
    text = render_markdown(res, prof)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str))
    print(text)
    print(f"written to {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
