from __future__ import annotations

from pathlib import Path

import pytest

from tmagent.config import Config, load_config

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_cfg(tmp_path: Path) -> Config:
    """configs/fake.yaml with tiny frames and the data root in tmp_path."""
    return load_config(
        REPO / "configs" / "fake.yaml",
        [
            f"data.root={(tmp_path / 'data').as_posix()}",
            "data.resolution=[32, 24]",
            "data.history_s=0.25",
        ],
    )
