"""Pure trajectory-geometry functionals for the offline floor tests (issues #7, #8).

These read the SHAPE of a hidden-state trajectory -- an axis the distributional
metrics (perplexity, surprisal, entropy) cannot reach. Every function here is pure
and numpy-only (no torch, no model): it takes arrays that the replay driver already
extracted and returns scalars, so it runs in milliseconds per turn on CPU and is
unit-testable without any model.

Definitions follow the issue specs:

* ``effective_dimension`` (d_rho) -- #7, Masoomi et al. 2026 (arXiv 2607.01571):
  the number of principal components of the trajectory covariance needed to capture
  ``rho`` of its variance.
* ``early_kinematics`` -- #7: seven summary features over the first fifth of the
  trajectory, treating the sequence of hidden states as a moving particle.
* ``mean_token_cosine`` (rho) -- #8, Bayat/Behrouz/Courville 2026 (arXiv 2606.23670):
  the per-token cosine between a layer's update and the residual entering it,
  averaged over tokens. rho_block / rho_MLP are the SAME function applied to the
  full-block update vs the MLP-alone update; the driver supplies the (update,
  residual) pairs, so the "which residual" choice lives at the call site, not baked
  into the math here.
* ``roc_auc`` -- rank-based (Mann-Whitney) AUC for scoring a signal against the
  pre-registered kill conditions. Tie-aware; no sklearn dependency.

Operationalization note (flagged for pre-registration sign-off): the vector
kinematics (mean_position, initial_state, final_state, mean_velocity) are returned
verbatim as d-vectors. HOW the seven features are combined into a single score for
an AUC (e.g. a regularized cross-validated linear probe vs per-feature) is a
separate research decision and is deliberately NOT fixed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_2d_float(a: object, *, name: str = "trajectory") -> Array:
    """Coerce to a 2-D (T, d) float64 array or raise with a clear message."""
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (T, d); got shape {x.shape}")
    return x


def effective_dimension(trajectory: object, rho: float) -> int:
    """d_rho: principal components of the trajectory covariance needed to capture
    ``rho`` of the variance.

    ``trajectory`` is (T, d): T successive hidden states (rows) in R^d. The result
    is order-invariant (covariance ignores row order). Returns 0 for a degenerate
    trajectory with no variance (fewer than 2 points, or all points identical).
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1]; got {rho}")
    x = _as_2d_float(trajectory)
    if x.shape[0] < 2:
        return 0
    xc = x - x.mean(axis=0, keepdims=True)
    # The covariance's eigenvalues equal the T x T Gram matrix's eigenvalues up to a
    # positive constant that cancels in the variance ratio. The Gram is T x T (cheap
    # when T << d) and symmetric PSD, so eigvalsh is stable -- this is issue #7's
    # "T x T Gram + eigen".
    gram = xc @ xc.T
    eig = np.linalg.eigvalsh(gram)  # ascending, may have tiny negatives from fp
    eig = np.clip(eig[::-1], 0.0, None)  # descending, non-negative
    total = float(eig.sum())
    if total <= 0.0:
        return 0
    cum = np.cumsum(eig) / total
    # First component whose cumulative variance ratio reaches rho (1-based count),
    # clamped to the number of meaningful (positive-eigenvalue) components so fp
    # error near rho == 1.0 cannot ask for more dimensions than the trajectory has.
    k = int(np.searchsorted(cum, rho, side="left")) + 1
    rank = int(np.count_nonzero(eig > total * 1e-12))
    return min(k, rank)


@dataclass(frozen=True)
class EarlyKinematics:
    """Seven early-kinematics features over the first fifth of a trajectory (#7).

    The vector features are full d-vectors; the scalar features are magnitudes. See
    the module docstring: collapsing these into a single AUC score is a separate,
    unfixed decision.
    """

    mean_position: Array
    positional_dispersion: float
    initial_state: Array
    final_state: Array
    mean_velocity: Array
    mean_speed: float
    speed_dispersion: float
    n_points: int  # size of the early window actually used (>= 2)


def early_kinematics(trajectory: object, fraction: float = 0.2) -> EarlyKinematics:
    """Compute the seven early kinematics over the first ``fraction`` of the
    trajectory (default 0.2 = the "first fifth" of #7).

    Treats the hidden states as a particle path: velocity is the per-step
    difference, speed its norm. Needs at least 2 positions (to define one velocity);
    the early window is at least 2 points even when ``fraction`` rounds smaller.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    x = _as_2d_float(trajectory)
    t = x.shape[0]
    if t < 2:
        raise ValueError(f"trajectory needs >= 2 positions for velocity; got {t}")
    k = min(t, max(2, int(np.ceil(t * fraction))))
    p = x[:k]
    v = np.diff(p, axis=0)  # (k-1, d) per-step velocity
    speeds = np.linalg.norm(v, axis=1)  # (k-1,)
    mean_pos = p.mean(axis=0)
    positional_dispersion = float(np.sqrt(np.mean(np.sum((p - mean_pos) ** 2, axis=1))))
    return EarlyKinematics(
        mean_position=mean_pos,
        positional_dispersion=positional_dispersion,
        initial_state=p[0].copy(),
        final_state=p[-1].copy(),
        mean_velocity=v.mean(axis=0),
        mean_speed=float(speeds.mean()),
        speed_dispersion=float(speeds.std()),
        n_points=k,
    )


def mean_token_cosine(update: object, residual: object, *, eps: float = 1e-12) -> float:
    """rho: per-token cosine between a layer's ``update`` and the ``residual``
    entering it, averaged over tokens (#8).

    Both arrays are (T, d) aligned by token. Tokens where either vector is ~zero
    have undefined cosine and are dropped; if none remain, returns 0.0. Applying
    this to the full-block update gives rho_block; to the MLP-alone update gives
    rho_MLP -- the driver chooses which residual each is measured against.
    """
    u = _as_2d_float(update, name="update")
    r = _as_2d_float(residual, name="residual")
    if u.shape != r.shape:
        raise ValueError(f"update and residual must share shape; got {u.shape} vs {r.shape}")
    denom = np.linalg.norm(u, axis=1) * np.linalg.norm(r, axis=1)
    valid = denom > eps
    if not np.any(valid):
        return 0.0
    dots = np.sum(u * r, axis=1)
    return float(np.mean(dots[valid] / denom[valid]))


def _average_ranks(a: Array) -> Array:
    """1-based ranks with ties assigned their group's average rank."""
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # mean of 1-based positions
        i = j
    return ranks


def roc_auc(scores: object, labels: object) -> float:
    """Rank-based (Mann-Whitney) ROC AUC: the probability a random positive scores
    above a random negative, with ties counted as half.

    ``labels`` are truthy (positive) / falsy (negative). Raises if either class is
    empty (AUC is undefined). Tie-aware, so all-equal scores give exactly 0.5.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError(f"scores and labels must be 1-D of equal length; got {s.shape}, {y.shape}")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs at least one positive and one negative label")
    ranks = _average_ranks(s)
    # Mann-Whitney U for the positive class -> AUC.
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
