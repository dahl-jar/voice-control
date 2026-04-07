"""Lazy keyboard backend helpers with platform-specific diagnostics."""

import os
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pynput.keyboard import Controller, Key


def format_keyboard_backend_error(exc: Exception) -> str:
    """Explain common platform-specific pynput failures."""
    base = "Keyboard control is unavailable because pynput could not start."

    if sys.platform.startswith("linux"):
        display = os.environ.get("DISPLAY")
        session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
        if not display:
            return (
                f"{base} Linux session type is '{session_type}' and DISPLAY is unset. "
                "pynput needs X11/Xwayland, or Linux uinput access as root. "
                f"Original error: {exc}"
            )
        return (
            f"{base} On Linux, pynput depends on X11/Xwayland or uinput access. "
            f"Original error: {exc}"
        )

    return f"{base} Original error: {exc}"


def load_keyboard_backend() -> tuple["Controller", type["Key"]]:
    """Import pynput lazily so unsupported platforms fail with a clear message."""
    try:
        from pynput.keyboard import Controller, Key
    except Exception as exc:
        raise RuntimeError(format_keyboard_backend_error(exc)) from exc

    return Controller(), Key
