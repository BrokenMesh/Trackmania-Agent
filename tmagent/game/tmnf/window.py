"""Bring the TMNF game window to the foreground (Windows only).

VERIFIED on TMUF 2.12 + TMI 2.2.1: after a `map` load the intro only finishes while
the game window has focus. Without focus the race start state is saved with the car
still locked by the intro, and every rewind to it drives nowhere (speed stays 0).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

WINDOW_TITLE_PREFIX = "TrackMania"
FOCUS_SETTLE_S = 0.3


def pid_listening_on(port: int) -> int | None:
    """PID of the process with a TCP listen socket on 127.0.0.1:<port> (netstat), or None.

    The state column is localized (LISTENING / ABHÖREN); listen sockets are matched by
    their remote address 0.0.0.0:0 instead.

    With several game instances this maps a plugin port to its game process.
    """
    if sys.platform != "win32":
        return None
    raw = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, check=False).stdout
    out = raw.decode("ascii", "replace")
    pattern = re.compile(rf"^\s*TCP\s+127\.0\.0\.1:{port}\s+0\.0\.0\.0:0\s+\S+\s+(\d+)\s*$")
    for line in out.splitlines():
        m = pattern.match(line)
        if m:
            return int(m.group(1))
    return None


def find_game_window(title_prefix: str = WINDOW_TITLE_PREFIX, pid: int | None = None) -> int | None:
    """Handle of the first visible top-level window whose title starts with the prefix
    (and that belongs to process `pid`, if given)."""
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
        if pid is not None and is_game:
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            is_game = owner.value == pid
        if is_game and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def focus_game_window(title_prefix: str = WINDOW_TITLE_PREFIX, pid: int | None = None) -> bool:
    """Restore and focus the game window; False if it was not found or focus was refused.

    Windows refuses SetForegroundWindow from a background process unless it received
    the last input event; a synthetic Alt press/release satisfies that rule.
    """
    hwnd = find_game_window(title_prefix, pid)
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


SCAN_ENTER = 0x1C


def press_key(scan_code: int, pid: int | None = None) -> bool:
    """Focus the game window (of `pid`, if given) and send one key press by scan code.

    The game reads the keyboard via DirectInput, so a posted window message is not
    enough; SendInput needs the window in the foreground.
    """
    if sys.platform != "win32" or not focus_game_window(pid=pid):
        return False
    import ctypes
    from ctypes import wintypes

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
        ]  # fmt: skip

    class INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]

        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    keyeventf_scancode, keyeventf_keyup, input_keyboard = 0x0008, 0x0002, 1
    for flags in (keyeventf_scancode, keyeventf_scancode | keyeventf_keyup):
        inp = INPUT(type=input_keyboard)
        inp.ki = KEYBDINPUT(0, scan_code, flags, 0, 0)
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.05)
    return True


VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 0x25, 0x26, 0x27, 0x28
VK_H = 0x48
DRIVE_KEYS = (VK_LEFT, VK_RIGHT, VK_UP, VK_DOWN)  # TMNF default binds: arrows


def keys_down(vks: tuple[int, ...]) -> tuple[bool, ...]:
    """Which of the virtual keys are held right now (GetAsyncKeyState, any window focused).
    All False off Windows."""
    if sys.platform != "win32":
        return tuple(False for _ in vks)
    import ctypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    return tuple(bool(user32.GetAsyncKeyState(vk) & 0x8000) for vk in vks)
