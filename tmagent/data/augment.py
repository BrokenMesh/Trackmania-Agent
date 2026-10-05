"""Training-time augmentation of one window sample (numpy dict, no batch dim).

Only the action history and the frames are touched; target, target_valid and
frame_valid are never modified. The input arrays are not mutated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tmagent.config import DataConfig
from tmagent.interfaces import BRAKE, GAS, STEER

MAX_SHIFT = 4  # px
JITTER = 0.1  # +-10% brightness / contrast


def augment_frames(
    frames: np.ndarray, valid: np.ndarray, rng: np.random.Generator, max_shift: int = MAX_SHIFT
) -> np.ndarray:
    """Brightness/contrast jitter + integer shift shared by all frames of a window.

    frames uint8 [K, C, H, W], valid bool [K]; padded (invalid) frames stay zero.
    The shift replicates edge pixels. Returns a new uint8 array.
    """
    if not valid.any():
        return np.zeros_like(frames)
    contrast, brightness = rng.uniform(1 - JITTER, 1 + JITTER, size=2)
    dy, dx = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
    mean = float(frames[valid].mean())
    levels = ((np.arange(256) - mean) * contrast + mean) * brightness
    lut = np.clip(np.rint(levels), 0, 255).astype(np.uint8)
    h, w = frames.shape[-2:]
    pad = ((0, 0), (0, 0), (max_shift, max_shift), (max_shift, max_shift))
    shifted = np.pad(frames, pad, mode="edge")[
        ..., max_shift + dy : max_shift + dy + h, max_shift + dx : max_shift + dx + w
    ]
    out = np.take(lut, shifted)  # new contiguous array
    out[~valid] = 0
    return out


def augment_sample(sample: dict[str, Any], cfg: DataConfig, rng: np.random.Generator) -> dict:
    """Augment the history (and frames if cfg.image_aug) of one window sample.

    With probability cfg.action_dropout the whole action history is marked
    invalid, otherwise each step is dropped with cfg.action_token_dropout. Steer
    history gets gaussian noise (std cfg.action_noise_steer, clipped), binary
    gas/brake history values flip with cfg.action_flip_prob, and hist_actions is
    zeroed wherever hist_valid is False.
    """
    out = dict(sample)
    hist_valid = np.array(sample["hist_valid"], dtype=bool)
    hist = np.array(sample["hist_actions"], dtype=np.float32)
    if rng.random() < cfg.action_dropout:
        hist_valid[:] = False
    elif cfg.action_token_dropout > 0:
        hist_valid &= rng.random(hist_valid.shape) >= cfg.action_token_dropout
    if cfg.action_noise_steer > 0:
        noise = rng.normal(0.0, cfg.action_noise_steer, size=hist.shape[:-1])
        hist[..., STEER] = np.clip(hist[..., STEER] + noise, -1.0, 1.0)
    if cfg.action_flip_prob > 0:
        for ch in (GAS, BRAKE):
            flip = rng.random(hist.shape[:-1]) < cfg.action_flip_prob
            hist[..., ch] = np.where(flip, 1.0 - hist[..., ch], hist[..., ch])
    hist[~hist_valid] = 0.0
    out["hist_valid"], out["hist_actions"] = hist_valid, hist
    if cfg.image_aug:
        frames = np.asarray(sample["frames"])
        out["frames"] = augment_frames(frames, np.asarray(sample["frame_valid"], bool), rng)
    return out
