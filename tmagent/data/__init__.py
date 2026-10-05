"""Data pipeline: input timelines, episode rendering and storage, splits, windows.

quality is intentionally not imported here (it is a `python -m` entry point).
WindowDataset is imported lazily so that light users do not pay for torch.
"""

from __future__ import annotations

from typing import Any

from tmagent.data.augment import augment_sample
from tmagent.data.episode_io import (
    EpisodeCache,
    append_index,
    load_episode,
    read_index,
    save_episode,
)
from tmagent.data.render import render_episode
from tmagent.data.split import filter_index, split_of
from tmagent.data.timeline import (
    control_times_ms,
    frame_times_ms,
    resample_to_control,
    timeline_from_events,
)

__all__ = [
    "EpisodeCache",
    "WindowDataset",
    "append_index",
    "augment_sample",
    "control_times_ms",
    "filter_index",
    "frame_times_ms",
    "load_episode",
    "read_index",
    "render_episode",
    "resample_to_control",
    "save_episode",
    "split_of",
    "timeline_from_events",
]


def __getattr__(name: str) -> Any:
    if name == "WindowDataset":
        from tmagent.data.dataset import WindowDataset

        return WindowDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
