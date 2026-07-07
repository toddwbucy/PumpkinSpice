"""Unit tests for the pure trajectory-geometry functionals (issues #7, #8).

No model or GPU: every case is a synthetic trajectory with a hand-computable
answer. Skips cleanly if the ``introspect`` extra (numpy) is not installed.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pumpkinspice.introspect import (  # noqa: E402  (after importorskip)
    early_kinematics,
    effective_dimension,
    mean_token_cosine,
    roc_auc,
)

# --- effective_dimension (d_rho) -------------------------------------------


def test_effective_dimension_rank_one_line() -> None:
    # All points on a single line -> one component captures everything.
    u = np.array([1.0, 2.0, 3.0])
    traj = np.outer(np.arange(4.0), u)  # 4 points along u
    for rho in (0.5, 0.75, 0.9, 1.0):
        assert effective_dimension(traj, rho) == 1


def test_effective_dimension_two_equal_axes() -> None:
    # Two orthogonal directions with equal variance: half the variance sits in
    # each, so rho <= 0.5 needs 1 component and rho > 0.5 needs 2.
    traj = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert effective_dimension(traj, 0.5) == 1
    assert effective_dimension(traj, 0.75) == 2
    assert effective_dimension(traj, 0.9) == 2


def test_effective_dimension_degenerate_cases() -> None:
    assert effective_dimension(np.zeros((5, 4)), 0.9) == 0  # no variance
    assert effective_dimension(np.ones((1, 4)), 0.9) == 0  # single point


def test_effective_dimension_rejects_bad_rho() -> None:
    traj = np.eye(3)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="rho"):
            effective_dimension(traj, bad)


def test_effective_dimension_requires_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        effective_dimension(np.arange(5.0), 0.9)


# --- early_kinematics -------------------------------------------------------


def test_early_kinematics_constant_velocity() -> None:
    step = np.array([3.0, 4.0])  # norm 5
    start = np.array([1.0, -2.0])
    traj = start + np.outer(np.arange(20.0), step)  # 20 points, constant velocity

    k = early_kinematics(traj, fraction=0.2)
    assert k.n_points == 4  # ceil(20 * 0.2)
    assert np.allclose(k.initial_state, start)
    assert np.allclose(k.final_state, start + 3 * step)
    assert np.allclose(k.mean_velocity, step)
    assert k.mean_speed == pytest.approx(5.0)
    assert k.speed_dispersion == pytest.approx(0.0)  # constant speed
    assert k.positional_dispersion > 0.0


def test_early_kinematics_window_floor_is_two() -> None:
    # A short trajectory still yields at least one velocity (>= 2 points).
    k = early_kinematics(np.zeros((3, 2)), fraction=0.01)
    assert k.n_points == 2


def test_early_kinematics_requires_two_points() -> None:
    with pytest.raises(ValueError, match="velocity"):
        early_kinematics(np.zeros((1, 4)))


def test_early_kinematics_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        early_kinematics(np.zeros((5, 2)), fraction=1.5)


# --- mean_token_cosine (rho) -----------------------------------------------


def test_mean_token_cosine_alignment_extremes() -> None:
    residual = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert mean_token_cosine(residual, residual) == pytest.approx(1.0)  # parallel
    assert mean_token_cosine(-residual, residual) == pytest.approx(-1.0)  # anti
    orth = np.array([[0.0, 1.0], [3.0, 0.0]])
    assert mean_token_cosine(orth, residual) == pytest.approx(0.0)  # orthogonal


def test_mean_token_cosine_drops_zero_rows() -> None:
    # One aligned token and one zero token -> average over the valid token only.
    update = np.array([[1.0, 0.0], [0.0, 0.0]])
    residual = np.array([[2.0, 0.0], [1.0, 1.0]])
    assert mean_token_cosine(update, residual) == pytest.approx(1.0)
    assert mean_token_cosine(np.zeros((2, 2)), residual) == 0.0  # nothing valid


def test_mean_token_cosine_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        mean_token_cosine(np.zeros((2, 2)), np.zeros((3, 2)))


# --- roc_auc ----------------------------------------------------------------


def test_roc_auc_perfect_and_reversed() -> None:
    labels = [0, 0, 1, 1]
    assert roc_auc([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(1.0)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(0.0)


def test_roc_auc_ties_give_half() -> None:
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_roc_auc_partial_with_a_tie() -> None:
    # neg=0.0, pos={0.0 (tie with neg), 1.0}: the tie counts as half.
    # U = (#pos>neg) + 0.5*(#pos==neg) = 1 + 0.5 = 1.5 over 1*2 pairs -> 0.75.
    assert roc_auc([0.0, 0.0, 1.0], [0, 1, 1]) == pytest.approx(0.75)


def test_roc_auc_single_class_raises() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        roc_auc([0.1, 0.2, 0.3], [1, 1, 1])


def test_roc_auc_shape_guard() -> None:
    with pytest.raises(ValueError, match="1-D of equal length"):
        roc_auc([0.1, 0.2, 0.3], [0, 1])  # length mismatch
    with pytest.raises(ValueError, match="1-D of equal length"):
        roc_auc([[0.1, 0.2]], [[0, 1]])  # 2-D
