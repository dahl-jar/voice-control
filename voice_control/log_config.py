"""
Project-wide logging setup, routed through rich.

One place to configure the root logger so every module writes through
the same rich-enabled handler. Entry points (inference.py, ui.py,
train.py, download_model.py, finetune.py, diagnostics.py) call
`configure_logging()` exactly once at startup. Library modules inside
the project should just do `logger = logging.getLogger(__name__)` and
use it — they must not configure handlers themselves, because doing
that from a library module would stomp whatever the entry point set up.

Level is controlled by the VOICE_CONTROL_LOG_LEVEL env var (default INFO).
Set it to DEBUG to see per-frame VAD traces and classification details
that would otherwise be gated off.

`get_console()` hands back the single `Console` instance that both the
logger and any `rich.live.Live` display write to. Sharing it matters:
if they were two different consoles, log lines would punch through the
live region instead of scrolling above it.
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler


DEFAULT_LEVEL = "INFO"
LEVEL_ENV_VAR = "VOICE_CONTROL_LOG_LEVEL"
_LOG_FORMAT = "%(name)s: %(message)s"
_TIME_FORMAT = "[%H:%M:%S]"

_console: Console | None = None


def get_console() -> Console:
    """Return the shared rich Console. Built lazily on first call."""
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def configure_logging() -> None:
    """Wire the root logger up to rich. Call this once, from an entry point."""
    level_name = os.environ.get(LEVEL_ENV_VAR, DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = RichHandler(
        console=get_console(),
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format=_TIME_FORMAT,
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
