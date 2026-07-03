"""Centralized logging setup.

Logs go to stderr so stdout stays clean for machine-readable command output
(e.g. the ``plugins`` listing). Use ``get_logger(__name__)`` in modules and call
``configure_logging`` once at process entry.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once. ``level`` falls back to $PUMPKINSPICE_LOG_LEVEL
    then INFO. Idempotent: safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        if level is not None:
            logging.getLogger("pumpkinspice").setLevel(_coerce_level(level))
        return
    # `is not None` (not truthiness): logging.NOTSET is 0, and an explicit 0
    # must not silently fall through to the environment/INFO default.
    resolved = _coerce_level(
        level if level is not None else (os.environ.get("PUMPKINSPICE_LOG_LEVEL") or "INFO")
    )
    logging.basicConfig(level=resolved, format=DEFAULT_FORMAT, datefmt="%H:%M:%S")
    _CONFIGURED = True


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
