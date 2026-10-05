"""Deterministic train/val/test split by map_uid (never by episode)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tmagent.config import DataConfig

SPLITS = ("train", "val", "test")
OVERRIDES_NAME = "splits.json"


def split_of(map_uid: str, seed: int, val_frac: float, test_frac: float) -> str:
    """Hash split: u = sha256(f"{seed}:{map_uid}") mapped to [0, 1).

    u < test_frac -> "test", u < test_frac + val_frac -> "val", else "train".
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac > 1:
        raise ValueError(f"invalid fractions val={val_frac} test={test_frac}")
    digest = hashlib.sha256(f"{seed}:{map_uid}".encode()).digest()
    u = (int.from_bytes(digest[:8], "big") >> 11) / 2**53  # 53 bits -> exact in [0, 1)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def load_overrides(root: str | Path | None) -> dict[str, str]:
    """{map_uid: split} from <root>/splits.json, empty if absent."""
    if root is None:
        return {}
    path = Path(root) / OVERRIDES_NAME
    if not path.exists():
        return {}
    overrides = json.loads(path.read_text())
    bad = {k: v for k, v in overrides.items() if v not in SPLITS}
    if bad:
        raise ValueError(f"{path}: invalid split names {bad}")
    return overrides


def split_for_map(map_uid: str, cfg: DataConfig, overrides: dict[str, str] | None = None) -> str:
    """Split of one map: override if present, else the hash split."""
    if overrides and map_uid in overrides:
        return overrides[map_uid]
    return split_of(map_uid, cfg.split_seed, cfg.val_frac, cfg.test_frac)


def filter_index(
    entries: Iterable[dict[str, Any]],
    split: str,
    cfg: DataConfig,
    root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Index entries whose map_uid belongs to `split` (splits.json in root overrides)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    overrides = load_overrides(root)
    return [e for e in entries if split_for_map(e["map_uid"], cfg, overrides) == split]
