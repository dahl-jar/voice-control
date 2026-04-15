"""
Project-wide logging setup.

One place to configure the root logger so every module writes through
the same format, level, and stream. Entry points (inference.py, ui.py,
train.py, download_model.py, finetune.py, diagnostics.py) call
`configure_logging()` exactly once at startup. Library modules inside
the project should just do `logger = logging.getLogger(__name__)` and
use it — they must not configure handlers themselves, because doing
that from a library module would stomp whatever the entry point set up.

Level is controlled by the VOICE_CONTROL_LOG_LEVEL env var (default INFO).
Set it to DEBUG to see per-frame VAD traces and classification details
that would otherwise be gated off.
"""

from __future__ import annotations

import logging
import os


DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
LEVEL_ENV_VAR = "VOICE_CONTROL_LOG_LEVEL"


def configure_logging() -> None:
    """
    Configure the root logger once per process. Safe to call multiple
    times — basicConfig is a no-op if handlers already exist, and the
    `force=True` path lets tests reset between runs if they need to.
    """
    level_name = os.environ.get(LEVEL_ENV_VAR, DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=False,
    )
