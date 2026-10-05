"""Torch dataset producing training windows (batch format: docs/ARCHITECTURE.md).

An item is (episode, k_end): a window of K = cfg.num_steps frame steps ending at
frame index k_end. For frame index fi (R = cfg.actions_per_frame):
  hist   = actions[fi*R - R : fi*R]          valid iff fi >= 1
  target = actions[fi*R : fi*R + chunk_len]  zero-padded, target_valid False past the end
Steps with fi < 0 are zero-padded with every valid flag False.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tmagent.config import DataConfig
from tmagent.data.augment import augment_sample
from tmagent.data.episode_io import EpisodeCache, EpisodeReader, read_index
from tmagent.data.split import filter_index
from tmagent.data.timeline import control_times_ms, frame_times_ms
from tmagent.interfaces import Episode


def check_alignment(ep: Episode | EpisodeReader, cfg: DataConfig, where: str = "episode") -> None:
    """Raise ValueError unless frames/actions sit on the exact formula grids."""
    r = cfg.actions_per_frame
    if isinstance(ep, Episode):
        t_f, shape = len(ep.frames), tuple(ep.frames.shape[1:])
    else:
        t_f, shape = ep.num_frames, ep.frame_shape
    t_a = len(ep.actions)
    if not np.array_equal(ep.frame_times_ms, frame_times_ms(t_f, cfg.frame_hz)):
        raise ValueError(f"{where}: frame_times_ms is not the {cfg.frame_hz} Hz grid")
    if not np.array_equal(ep.action_times_ms, control_times_ms(t_a, cfg.control_hz)):
        raise ValueError(f"{where}: action_times_ms is not the {cfg.control_hz} Hz grid")
    if t_f and t_a < (t_f - 1) * r + 1:
        raise ValueError(f"{where}: {t_a} actions too few for {t_f} frames (R={r})")
    w, h = cfg.resolution
    if t_f and shape != (h, w, cfg.channels):
        raise ValueError(f"{where}: frames {shape} != {(h, w, cfg.channels)}")


def build_window(ep: Episode | EpisodeReader, k_end: int, cfg: DataConfig) -> dict[str, np.ndarray]:
    """Numpy window ending at frame index k_end (no augmentation).

    With an EpisodeReader only the window's valid frames are read and decoded.
    """
    k, r, c_len = cfg.num_steps, cfg.actions_per_frame, cfg.chunk_len
    t_a = len(ep.actions)
    fi = np.arange(k_end - k + 1, k_end + 1)
    frame_valid = fi >= 0
    fc = np.maximum(fi, 0)

    h, w, c = ep.frame_shape if isinstance(ep, EpisodeReader) else ep.frames.shape[1:]
    valid_idx = fi[frame_valid]
    imgs = ep.frames(valid_idx) if isinstance(ep, EpisodeReader) else ep.frames[valid_idx]
    frames = np.zeros((k, c, h, w), dtype=np.uint8)
    frames[k - len(valid_idx) :] = imgs.transpose(0, 3, 1, 2)  # padded steps come first

    hist_valid = fi >= 1
    hidx = np.clip(fc[:, None] * r - r + np.arange(r), 0, t_a - 1)
    hist = ep.actions[hidx].astype(np.float32)
    hist[~hist_valid] = 0.0

    tidx = fc[:, None] * r + np.arange(c_len)
    target_valid = frame_valid[:, None] & (tidx < t_a)
    target = ep.actions[np.minimum(tidx, t_a - 1)].astype(np.float32)
    target[~target_valid] = 0.0
    return {
        "frames": frames,
        "frame_valid": frame_valid,
        "hist_actions": hist,
        "hist_valid": hist_valid,
        "target": target,
        "target_valid": target_valid,
    }


class WindowDataset(Dataset):
    """All windows (episode, k_end), k_end = 0, stride, 2*stride, ... per episode.

    Episode readers (cheap, no frames) are cached per process and each item reads
    and decodes only its K frames from disk. The augmentation RNG is created per
    process (seeded from torch.initial_seed(), which differs per DataLoader
    worker), so the dataset is safe with num_workers > 0 and holds no open files.
    """

    def __init__(
        self,
        root: str,
        cfg: DataConfig,
        split: str,
        train: bool,
        stride: int = 1,
        cache_size: int = 32,
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if cfg.control_hz % cfg.frame_hz != 0:
            raise ValueError("control_hz must be a multiple of frame_hz")
        self.root = str(root)
        self.cfg = cfg
        self.split = split
        self.train = train
        self.stride = stride
        self.cache_size = cache_size
        self.entries = filter_index(read_index(self.root), split, cfg, self.root)
        for e in self.entries:
            self._check_entry(e)
        counts = [-(-int(e["num_frames"]) // stride) for e in self.entries]
        self._starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._pid: int | None = None
        self._cache: EpisodeCache | None = None
        self._rng: np.random.Generator | None = None

    def _check_entry(self, e: dict[str, Any]) -> None:
        want = {
            "frame_hz": self.cfg.frame_hz,
            "control_hz": self.cfg.control_hz,
            "resolution": list(self.cfg.resolution),
            "channels": self.cfg.channels,
        }
        bad = {k: (e[k], v) for k, v in want.items() if k in e and e[k] != v}
        if bad:
            raise ValueError(f"{e['path']}: index disagrees with config (index, cfg): {bad}")

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_pid=None, _cache=None, _rng=None)  # never ship cache/RNG to workers
        return state

    def _load(self, path: Path) -> EpisodeReader:
        reader = EpisodeReader(path)
        check_alignment(reader, self.cfg, str(path))
        return reader

    def _process_state(self) -> tuple[EpisodeCache, np.random.Generator]:
        if self._pid != os.getpid() or self._cache is None or self._rng is None:
            self._pid = os.getpid()
            self._cache = EpisodeCache(self.cache_size, loader=self._load)
            self._rng = np.random.default_rng(torch.initial_seed())
        return self._cache, self._rng

    def locate(self, idx: int) -> tuple[int, int]:
        """(entry index, k_end) of item idx."""
        e = int(np.searchsorted(self._starts, idx, side="right")) - 1
        return e, (idx - int(self._starts[e])) * self.stride

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self)
        if not -n <= idx < n:
            raise IndexError(f"index {idx} out of range for {n} windows")
        e, k_end = self.locate(idx % n)
        cache, rng = self._process_state()
        entry = self.entries[e]
        reader = cache.get(Path(self.root) / entry["path"])
        if reader.num_frames != entry["num_frames"]:
            raise ValueError(f"{entry['path']}: index num_frames != {reader.num_frames} in file")
        sample = build_window(reader, k_end, self.cfg)
        if self.train:
            sample = augment_sample(sample, self.cfg, rng)
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sample.items()}
