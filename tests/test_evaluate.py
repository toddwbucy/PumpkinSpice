"""Tests for the floor-test evaluator (issues #7, #8).

Synthetic LabeledTurns with a known separating structure: the AUCs must be high when
a feature tracks the label and ~0.5 (or a KILL) when it does not. Offline; skips if
the ``evaluate`` extra (numpy + scikit-learn) is absent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

import numpy as np

from pumpkinspice.introspect.evaluate import (
    LabeledTurn,
    evaluate_floor_test,
    kinematic_features,
    labeled_turn_from_dict,
    labeled_turn_to_dict,
    load_labeled_turns,
)
from pumpkinspice.introspect.geometry import EarlyKinematics
from pumpkinspice.introspect.replay import TrajectoryMetrics


def _metrics(
    *,
    mean_speed: float,
    drho: tuple[int, int, int] = (1, 2, 3),
    n_layers: int = 4,
    rho_block: list[float] | None = None,
) -> TrajectoryMetrics:
    d = 4
    kin = EarlyKinematics(
        mean_position=np.zeros(d),
        positional_dispersion=0.0,
        initial_state=np.zeros(d),
        final_state=np.zeros(d),
        mean_velocity=np.zeros(d),
        mean_speed=mean_speed,
        speed_dispersion=0.0,
        n_points=2,
    )
    block = np.asarray(
        rho_block if rho_block is not None else list(np.linspace(1.0, 0.0, n_layers))
    )
    return TrajectoryMetrics(
        d_rho={0.5: drho[0], 0.75: drho[1], 0.9: drho[2]},
        kinematics=kin,
        rho_block=block,
        rho_mlp=np.zeros(n_layers),
        n_prompt_tokens=3,
        n_output_tokens=5,
        n_layers=n_layers,
    )


def _corpus(
    task_type: str, n: int, *, correct_fast: bool = True, seed: int = 0
) -> list[LabeledTurn]:
    """n turns where correctness tracks mean_speed (direction set by correct_fast)
    and difficulty tracks d_rho (hard -> higher)."""
    rng = np.random.default_rng(seed)
    turns = []
    for i in range(n):
        correct = i % 2 == 0
        hard = bool(rng.random() < 0.5)  # independent of correctness
        fast = correct if correct_fast else not correct
        mean_speed = float(rng.normal(5.0 if fast else 1.0, 0.3))
        drho = (int(rng.integers(4, 6)),) * 3 if hard else (int(rng.integers(1, 3)),) * 3
        turns.append(
            LabeledTurn(task_type, correct, hard, _metrics(mean_speed=mean_speed, drho=drho))
        )  # type: ignore[arg-type]
    return turns


def test_kinematics_and_drho_aucs_are_high_when_separable() -> None:
    report = evaluate_floor_test(_corpus("tool_use", 60))
    assert report.n_by_type == {"tool_use": 60}
    # correctness tracks mean_speed -> kinematics probe separates well
    assert report.kinematics_correctness["tool_use"] is not None
    assert report.kinematics_correctness["tool_use"] > 0.85
    # difficulty tracks d_rho -> d_rho probe separates well
    assert report.drho_hard_easy["tool_use"] is not None
    assert report.drho_hard_easy["tool_use"] > 0.85
    # kill #1 applies to the agentic type and should PASS here
    kill1 = next(k for k in report.kills_hash7 if k.name.startswith("kill1"))
    assert kill1.passed is True


def test_cross_transfer_passes_when_direction_shared() -> None:
    turns = _corpus("tool_use", 40, correct_fast=True, seed=1) + _corpus(
        "reasoning", 40, correct_fast=True, seed=2
    )
    report = evaluate_floor_test(turns)
    for direction in ("tool_use->reasoning", "reasoning->tool_use"):
        assert report.cross_transfer[direction] is not None
        assert report.cross_transfer[direction] > 0.7


def test_cross_transfer_kill_when_direction_flips() -> None:
    # reasoning encodes correctness with the OPPOSITE speed sign -> a probe trained on
    # one type mis-ranks the other, so transfer collapses below 0.5 and kill #3 fires.
    turns = _corpus("tool_use", 40, correct_fast=True, seed=3) + _corpus(
        "reasoning", 40, correct_fast=False, seed=4
    )
    report = evaluate_floor_test(turns)
    assert report.cross_transfer["tool_use->reasoning"] < 0.5
    kill3 = next(k for k in report.kills_hash7 if k.name == "kill3_transfer[tool_use->reasoning]")
    assert kill3.passed is False


def test_undefined_auc_when_single_class() -> None:
    # every turn correct -> the correctness probe is undefined, not a crash.
    turns = [
        LabeledTurn("reasoning", True, i % 2 == 0, _metrics(mean_speed=float(i))) for i in range(10)
    ]
    report = evaluate_floor_test(turns)
    assert report.kinematics_correctness["reasoning"] is None
    kill2 = next(k for k in report.kills_hash7 if k.name.startswith("kill2"))
    assert kill2.passed is None
    assert "UNDEFINED" in report.summary()


def test_rho_curve_flatness_flag() -> None:
    flat = [LabeledTurn("t", True, False, _metrics(mean_speed=1.0, rho_block=[0.5, 0.5, 0.5, 0.5]))]
    structured = [
        LabeledTurn("t", True, False, _metrics(mean_speed=1.0, rho_block=[0.9, 0.6, 0.3, 0.0]))
    ]
    assert evaluate_floor_test(flat).rho_curves.flat is True
    assert evaluate_floor_test(structured).rho_curves.flat is False


def test_kinematic_features_layout() -> None:
    m = _metrics(mean_speed=3.0)
    feats = kinematic_features(m)
    assert feats.shape == (7,)
    assert feats[5] == pytest.approx(3.0)  # mean_speed slot


def test_serialization_round_trip() -> None:
    turn = _corpus("tool_use", 2)[0]
    back = labeled_turn_from_dict(labeled_turn_to_dict(turn))
    assert back.task_type == turn.task_type
    assert back.correct == turn.correct and back.hard == turn.hard
    assert back.metrics.d_rho == turn.metrics.d_rho
    assert np.allclose(back.metrics.rho_block, turn.metrics.rho_block)
    assert back.metrics.kinematics.mean_speed == turn.metrics.kinematics.mean_speed


def test_load_labeled_turns_and_cli(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pumpkinspice.cli import main

    path = tmp_path / "labeled.jsonl"
    turns = _corpus("tool_use", 20) + _corpus("reasoning", 20, seed=9)
    path.write_text("\n".join(json.dumps(labeled_turn_to_dict(t)) for t in turns))

    assert len(load_labeled_turns(path)) == 40

    out = tmp_path / "report.json"
    rc = main(["floortest", str(path), "--json", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["n_by_type"] == {"tool_use": 20, "reasoning": 20}
    assert any(k["name"].startswith("kill1") for k in report["kills_hash7"])


def test_empty_turns_raises() -> None:
    with pytest.raises(ValueError, match="no turns"):
        evaluate_floor_test([])
