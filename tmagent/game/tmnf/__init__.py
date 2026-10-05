"""TMNF bridge: TrackMania Nations Forever through the TMAgentLink TMInterface plugin.

Importing this package pulls in nothing heavy: pygbx (replay files) is imported only
inside `replay.load_replay`, and the public classes below load on first access.

    from tmagent.game.tmnf import TMNFGame            # SyncGame + RealtimeGame
    from tmagent.game.tmnf import FakePluginServer    # protocol fake for tests/smoke

Protocol: PROTOCOL.md next to this file. Setup: docs/setup_windows.md.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tmagent.game.tmnf.client import TMAgentClient
    from tmagent.game.tmnf.fake_server import FakePluginServer
    from tmagent.game.tmnf.game import TMNFGame

_LAZY = {
    "TMNFGame": "game",
    "TMAgentClient": "client",
    "FakePluginServer": "fake_server",
}

__all__ = ["FakePluginServer", "TMAgentClient", "TMNFGame"]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
