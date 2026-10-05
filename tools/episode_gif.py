"""Rendered episodes -> animated GIFs with the recorded inputs drawn in (needs Pillow).

    python tools/episode_gif.py data/tmnf/episodes/<map_uid>/<episode_id>
    python tools/episode_gif.py --config configs/tmnf.yaml --map A01 --limit 3

Overlay: steering bar (left/right), gas (green) and brake (red) lamps, speed and race
time. The action shown with a frame is the one held at that frame's race time. GIFs go
to --out (default <data.root>/gifs) as <episode_id>.gif.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmagent.config import load_config  # noqa: E402
from tmagent.data.episode_io import load_episode, read_index  # noqa: E402
from tmagent.interfaces import Episode  # noqa: E402

SCALE = 4
BAR_H = 28
GREEN, RED, GREY = (40, 200, 70), (220, 50, 50), (70, 70, 70)
WHITE, STEER = (240, 240, 240), (255, 170, 40)


def action_rows_for_frames(ep: Episode) -> np.ndarray:
    """Index of the action held at each frame time (last action time <= frame time)."""
    rows = np.searchsorted(ep.action_times_ms, ep.frame_times_ms, side="right") - 1
    return np.clip(rows, 0, len(ep.actions) - 1)


def draw_overlay(img, steer: float, gas: float, brake: float, speed: float, time_ms: int):
    from PIL import Image, ImageDraw

    w, h = img.size
    canvas = Image.new("RGB", (w, h + BAR_H), (20, 20, 20))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    y0, y1 = h + 6, h + BAR_H - 6
    mid, half = w // 2, w // 4
    d.rectangle([mid - half, y0, mid + half, y1], outline=GREY)
    x = mid + int(steer * half)
    d.rectangle([min(mid, x), y0, max(mid, x), y1], fill=STEER)
    d.line([mid, y0 - 2, mid, y1 + 2], fill=WHITE)
    d.ellipse([8, y0, 8 + (y1 - y0), y1], fill=GREEN if gas >= 0.5 else GREY)
    d.ellipse([30, y0, 30 + (y1 - y0), y1], fill=RED if brake >= 0.5 else GREY)
    d.text((mid + half + 8, y0), f"{speed:5.0f} km/h", fill=WHITE)
    d.text((56, y0), f"{time_ms / 1000:6.2f} s", fill=WHITE)
    return canvas


def episode_to_gif(ep: Episode, out: Path, scale: int = SCALE) -> Path:
    from PIL import Image

    rows = action_rows_for_frames(ep)
    images = []
    for frame, row, t in zip(ep.frames, rows, ep.frame_times_ms, strict=True):
        img = Image.fromarray(frame if frame.shape[-1] == 3 else frame[..., 0])
        img = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.NEAREST)
        steer, gas, brake = (float(v) for v in ep.actions[row])
        images.append(draw_overlay(img, steer, gas, brake, float(ep.speeds_kmh[row]), int(t)))
    frame_ms = int(np.median(np.diff(ep.frame_times_ms))) if len(ep.frame_times_ms) > 1 else 50
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=frame_ms, loop=0)
    return out


def select_episodes(root: Path, map_ref: str | None, limit: int | None) -> list[Path]:
    """Episode dirs from the index, filtered by (part of) map name or uid."""
    paths = []
    for e in read_index(root):
        names = [str(e.get(k, "")) for k in ("map_uid", "map_name", "episode_id")]
        if map_ref is None or any(map_ref.lower() in n.lower() for n in names):
            paths.append(root / e["path"])
    return paths[:limit] if limit else paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("episodes", nargs="*", help="episode directories")
    ap.add_argument("--config", default="configs/tmnf.yaml", help="data.root holds the index")
    ap.add_argument("--map", default=None, help="episodes whose map name/uid/id contains this")
    ap.add_argument("--limit", type=int, default=None, help="at most N episodes")
    ap.add_argument("--out", default=None, help="output directory (default <data.root>/gifs)")
    ap.add_argument("--scale", type=int, default=SCALE)
    args = ap.parse_args(argv)

    root = Path(load_config(args.config).data.root)
    paths = [Path(p) for p in args.episodes] or select_episodes(root, args.map, args.limit)
    if not paths:
        print("no episodes found")
        return 1
    out_dir = Path(args.out) if args.out else root / "gifs"
    for path in paths:
        ep = load_episode(path)
        name = ep.meta.get("episode_id", path.name)
        gif = episode_to_gif(ep, out_dir / f"{name}.gif", args.scale)
        print(f"{gif}  ({len(ep.frames)} frames, {ep.meta.get('race_time_ms')} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
