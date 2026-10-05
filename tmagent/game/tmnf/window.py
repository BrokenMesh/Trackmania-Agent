"""Bring the TMNF game window to the foreground (Windows only).

VERIFIED on TMUF 2.12 + TMI 2.2.1: after a `map` load the intro only finishes while
the game window has focus. Without focus the race start state is saved with the car
still locked by the intro, and every rewind to it drives nowhere (speed stays 0).
"""

from __future__ import annotations

import sys
import time

WINDOW_TITLE_PREFIX = "TrackMania"
FOCUS_SETTLE_S = 0.3


def find_game_window(title_prefix: str = WINDOW_TITLE_PREFIX) -> int | None:
    """Handle of the first visible top-level window whose title starts with the prefix."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value
        is_game = title.startswith(title_prefix) and "ModLoader" not in title
        if is_game and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def focus_game_window(title_prefix: str = WINDOW_TITLE_PREFIX) -> bool:
    """Restore and focus the game window; False if it was not found or focus was refused.

    Windows refuses SetForegroundWindow from a background process unless it received
    the last input event; a synthetic Alt press/release satisfies that rule.
    """
    hwnd = find_game_window(title_prefix)
    if hwnd is None:
        return False
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if user32.GetForegroundWindow() == hwnd:
        return True

    vk_menu, keyup = 0x12, 0x0002
    user32.keybd_event(vk_menu, 0, 0, 0)
    user32.keybd_event(vk_menu, 0, keyup, 0)

    sw_restore = 9
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, sw_restore)
    user32.SetForegroundWindow(hwnd)
    time.sleep(FOCUS_SETTLE_S)
    return user32.GetForegroundWindow() == hwnd
