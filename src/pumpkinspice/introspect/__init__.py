"""Offline model-introspection metrics for the trajectory-geometry floor tests
(issues #7 and #8).

This subpackage is the analysis side of the confirmation testbed: it never touches
the live decoder. ``geometry`` holds the pure, numpy-only functionals; a separate
replay driver (added later, behind the ``replay`` extra) supplies the hidden-state
arrays by teacher-forced replay of recorded turns.
"""

from __future__ import annotations

from pumpkinspice.introspect.geometry import (
    EarlyKinematics,
    early_kinematics,
    effective_dimension,
    mean_token_cosine,
    roc_auc,
)

__all__ = [
    "EarlyKinematics",
    "early_kinematics",
    "effective_dimension",
    "mean_token_cosine",
    "roc_auc",
]
