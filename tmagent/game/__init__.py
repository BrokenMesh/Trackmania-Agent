"""Game backends. `make_game` picks one from `GameConfig.backend`."""

from __future__ import annotations

from tmagent.config import DataConfig, GameConfig
from tmagent.game.fake import FakeGame
from tmagent.interfaces import RealtimeGame, SyncGame


def make_game(game_cfg: GameConfig, data_cfg: DataConfig) -> SyncGame | RealtimeGame:
    """Build the backend named by `game_cfg.backend` ("fake" or "tmnf")."""
    if game_cfg.backend == "fake":
        return FakeGame(game_cfg, data_cfg)
    if game_cfg.backend == "tmnf":
        from tmagent.game.tmnf import TMNFGame  # lazy: needs Windows-only dependencies

        return TMNFGame(game_cfg, data_cfg)
    raise ValueError(f"unknown game backend {game_cfg.backend!r} (expected 'fake' or 'tmnf')")


__all__ = ["FakeGame", "make_game"]
