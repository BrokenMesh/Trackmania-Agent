"""Episode storage (format v2) and the index.jsonl catalogue.

Layout: <root>/episodes/<map_uid>/<episode_id>/ holding
  frames.bin   per-frame zlib blobs (level 3) of uint8 (H, W, C) frames, concatenated
  arrays.npz   frame_offsets [T_f+1], frame_times_ms, actions, action_times_ms,
               positions, speeds_kmh, frame_shape [H, W, C]  (uncompressed)
  meta.json    episode meta, incl. "format": 2
and <root>/index.jsonl (one line per episode: meta + "path" of the episode
directory relative to root + "num_frames").

Episode directories are written to <episode_id>.tmp and renamed, index lines
are appended with a single O_APPEND write, so readers never see partial
episodes and concurrent writers do not interleave lines. EpisodeReader decodes
frames lazily by random access, holding no open file between calls, so it is
safe to use from forked DataLoader workers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import warnings
import zlib
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.interfaces import Episode

FORMAT = 2
INDEX_NAME = "index.jsonl"
_SMALL_ARRAYS = ("frame_times_ms", "actions", "action_times_ms", "positions", "speeds_kmh")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _safe(name: str) -> str:
    """File-system safe path component."""
    out = re.sub(r"[^A-Za-z0-9._-]", "_", str(name)).lstrip(".")
    if not out:
        raise ValueError(f"empty path component from {name!r}")
    return out


def episode_path(root: str | Path, map_uid: str, episode_id: str) -> Path:
    """<root>/episodes/<map_uid>/<episode_id> (the episode directory, components sanitized)."""
    return Path(root) / "episodes" / _safe(map_uid) / _safe(episode_id)


def _write_dir(d: Path, ep: Episode) -> None:
    if ep.frames.ndim != 4 or ep.frames.dtype != np.uint8:
        raise ValueError(
            f"frames must be uint8 (T, H, W, C), got {ep.frames.dtype} {ep.frames.shape}"
        )
    offsets = np.zeros(len(ep.frames) + 1, dtype=np.int64)
    with open(d / "frames.bin", "wb") as f:
        for i, frame in enumerate(ep.frames):
            offsets[i + 1] = offsets[i] + f.write(zlib.compress(frame.tobytes(), level=3))
    arrays = {k: getattr(ep, k) for k in _SMALL_ARRAYS}
    np.savez(
        d / "arrays.npz",
        frame_offsets=offsets,
        frame_shape=np.array(ep.frames.shape[1:], dtype=np.int64),
        **arrays,
    )
    meta = {**ep.meta, "format": FORMAT}
    (d / "meta.json").write_text(json.dumps(meta, default=_json_default))


def save_episode(ep: Episode, root: str | Path) -> Path:
    """Write the episode directory atomically (tmp dir + rename); return its path."""
    path = episode_path(root, ep.meta["map_uid"], ep.meta["episode_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp, old = path.with_name(f"{path.name}.tmp"), path.with_name(f"{path.name}.old")
    for stale in (tmp, old):
        shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir()
    moved = False
    try:
        _write_dir(tmp, ep)
        if path.exists():  # overwrite: park the old directory, restore it on failure
            os.rename(path, old)
            moved = True
        os.rename(tmp, path)
    except BaseException:
        if moved:
            os.rename(old, path)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    return path


class EpisodeReader:
    """Lazy reader of one episode directory.

    The small arrays and the meta are loaded eagerly; frames are decoded on
    demand with frames(indices). No file stays open between calls.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._meta: dict[str, Any] = json.loads((self.path / "meta.json").read_text())
        if self._meta.get("format") != FORMAT:
            raise ValueError(
                f"{self.path}: unsupported episode format {self._meta.get('format')!r}"
            )
        with np.load(self.path / "arrays.npz", allow_pickle=False) as z:
            self._arrays = {k: z[k] for k in (*_SMALL_ARRAYS, "frame_offsets", "frame_shape")}
        self._offsets: np.ndarray = self._arrays["frame_offsets"]
        self.frame_shape: tuple[int, ...] = tuple(int(v) for v in self._arrays["frame_shape"])

    meta = property(lambda self: self._meta)
    frame_times_ms = property(lambda self: self._arrays["frame_times_ms"])
    actions = property(lambda self: self._arrays["actions"])
    action_times_ms = property(lambda self: self._arrays["action_times_ms"])
    positions = property(lambda self: self._arrays["positions"])
    speeds_kmh = property(lambda self: self._arrays["speeds_kmh"])

    @property
    def num_frames(self) -> int:
        return len(self._offsets) - 1

    def frames(self, indices: Any) -> np.ndarray:
        """Decode frames at `indices` -> uint8 [n, H, W, C] (runs of consecutive frames
        are fetched with one read)."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        out = np.empty((len(idx), *self.frame_shape), dtype=np.uint8)
        if len(idx) == 0:
            return out
        if idx.min() < 0 or idx.max() >= self.num_frames:
            raise IndexError(f"frame index out of range [0, {self.num_frames}): {idx}")
        uniq, inverse = np.unique(idx, return_inverse=True)
        dec = np.empty((len(uniq), *self.frame_shape), dtype=np.uint8)
        off, size = self._offsets, int(np.prod(self.frame_shape))
        starts = np.concatenate([[0], np.flatnonzero(np.diff(uniq) != 1) + 1, [len(uniq)]])
        with open(self.path / "frames.bin", "rb") as f:
            for a, b in zip(starts[:-1], starts[1:], strict=True):
                lo, hi = int(uniq[a]), int(uniq[b - 1])
                f.seek(int(off[lo]))
                blob = memoryview(f.read(int(off[hi + 1] - off[lo])))
                if len(blob) != off[hi + 1] - off[lo]:
                    raise ValueError(f"{self.path}: frames.bin is truncated")
                for j, i in enumerate(uniq[a:b], start=a):
                    chunk = blob[int(off[i] - off[lo]) : int(off[i + 1] - off[lo])]
                    try:
                        raw = zlib.decompress(chunk)
                    except zlib.error as exc:
                        raise ValueError(f"{self.path}: corrupt frame {i}: {exc}") from exc
                    if len(raw) != size:
                        raise ValueError(
                            f"{self.path}: frame {i} has {len(raw)} bytes, want {size}"
                        )
                    dec[j] = np.frombuffer(raw, dtype=np.uint8).reshape(self.frame_shape)
        return dec if len(uniq) == len(idx) and np.array_equal(uniq, idx) else dec[inverse]

    def to_episode(self) -> Episode:
        """Decode everything into an in-memory Episode."""
        return Episode(
            frames=self.frames(np.arange(self.num_frames)),
            meta=dict(self._meta),
            **{k: self._arrays[k] for k in _SMALL_ARRAYS},
        )


def load_episode(path: str | Path) -> Episode:
    """Read a whole episode directory (decodes all frames)."""
    return EpisodeReader(path).to_episode()


def append_index(root: str | Path, ep: Episode, path: str | Path) -> None:
    """Append one index line: meta + "path" (relative to root) + "num_frames".

    `path` is the episode directory returned by save_episode(ep, root).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rel = Path(os.path.relpath(path, root)).as_posix()
    entry = {**ep.meta, "format": FORMAT, "path": rel, "num_frames": int(len(ep.frames))}
    line = (json.dumps(entry, default=_json_default) + "\n").encode()
    fd = os.open(root / INDEX_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_index(root: str | Path) -> list[dict[str, Any]]:
    """All index entries (empty list if there is no index).

    Re-indexed episodes (same path) keep only their last entry. Unparseable
    lines (e.g. a torn write) are skipped with a warning.
    """
    index = Path(root) / INDEX_NAME
    if not index.exists():
        return []
    by_path: dict[str, dict[str, Any]] = {}
    for lineno, line in enumerate(index.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            by_path[entry["path"]] = entry
        except (ValueError, KeyError, TypeError):
            warnings.warn(f"{index}:{lineno}: skipping unreadable index line", stacklevel=2)
    return list(by_path.values())


class EpisodeCache:
    """Small LRU cache of EpisodeReaders keyed by path (readers are cheap: no frames)."""

    def __init__(
        self, size: int = 32, loader: Callable[[Path], EpisodeReader] = EpisodeReader
    ) -> None:
        self.size = max(int(size), 1)
        self._loader = loader
        self._items: OrderedDict[str, EpisodeReader] = OrderedDict()

    def get(self, path: str | Path) -> EpisodeReader:
        key = str(path)
        reader = self._items.get(key)
        if reader is None:
            reader = self._loader(Path(path))
            self._items[key] = reader
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        else:
            self._items.move_to_end(key)
        return reader

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, path: object) -> bool:
        return str(path) in self._items
