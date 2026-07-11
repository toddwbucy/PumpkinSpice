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
    dtype: str = "bfloat16",
    n_output_tokens: int = 5,
    trajectory_span: str = "output",
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
        n_output_tokens=n_output_tokens,
        n_layers=n_layers,
        dtype=dtype,
        trajectory_span=trajectory_span,
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
        )
    return turns


def _reasoning_levels_corpus(
    *, per_level_questions: int = 4, samples_per_q: int = 5, seed: int = 0
) -> list[LabeledTurn]:
    """MATH-shaped: questions at levels 1/3/5, several trajectories each (grouped by question),
    with d_rho tracking level (5 high, 3 medium, 1 low). The 1-vs-5 probe must separate the
    extremes with level 3 excluded."""
    rng = np.random.default_rng(seed)
    turns: list[LabeledTurn] = []
    q = 0
    for level in (1, 3, 5):
        base = {1: 1, 3: 3, 5: 5}[level]
        for _ in range(per_level_questions):
            q += 1
            for _s in range(samples_per_q):
                dval = int(rng.integers(base, base + 2))
                turns.append(
                    LabeledTurn(
                        "reasoning",
                        bool(rng.random() < 0.5),
                        level >= 4,
                        _metrics(mean_speed=1.0, drho=(dval, dval, dval)),
                        group=f"q{q}",
                        level=level,
                    )
                )
    return turns


def test_drho_1v5_difficulty_probe() -> None:
    from pumpkinspice.introspect.evaluate import report_to_dict

    report = evaluate_floor_test(_reasoning_levels_corpus(), difficulty_levels=(1, 5))
    # d_rho separates the 1-vs-5 extremes (level 3 excluded), leave-one-question-out
    assert report.drho_1v5["reasoning"] is not None and report.drho_1v5["reasoning"] > 0.85
    # registered as the reasoning-arm kill #1
    k = next(k for k in report.kills_hash7 if k.name == "kill1_drho_1v5[reasoning]")
    assert k.passed is True
    # the length control on the 1-vs-5 subset is reported, and drho_1v5 serializes
    assert "difficulty_1v5" in report.length_control["reasoning"]
    assert "drho_1v5" in report_to_dict(report)


def test_drho_1v5_absent_without_levels() -> None:
    # the agentic arm has no per-turn level (0) -> no 1-vs-5 probe, no reasoning kill1
    report = evaluate_floor_test(_corpus("tool_use", 40))
    assert report.drho_1v5["tool_use"] is None
    assert not any(k.name.startswith("kill1_drho_1v5") for k in report.kills_hash7)


def test_labeled_turn_serializes_level() -> None:
    t = LabeledTurn("reasoning", True, False, _metrics(mean_speed=1.0), group="q1", level=5)
    d = labeled_turn_to_dict(t)
    assert d["level"] == 5 and labeled_turn_from_dict(d).level == 5
    d.pop("level")  # a pre-1v5 metrics row has no level -> 0
    assert labeled_turn_from_dict(d).level == 0


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
    assert back.metrics.dtype == turn.metrics.dtype  # provenance survives serialization


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
    # length-confound control is serialized, nested by task type then kind
    assert set(report["length_control"]["reasoning"]) == {"correctness", "difficulty"}
    assert set(report["length_control"]["reasoning"]["correctness"]) == {
        "geometry_auc",
        "length_auc",
        "combined_auc",
        "marginal",
    }
    assert "reasoning->tool_use" in report["cross_transfer_length"]  # kill3 length baseline


def test_length_control_isolates_the_length_confound() -> None:
    # Correctness tracks generation LENGTH (n_output_tokens); the geometry (mean_speed)
    # is pure noise. The length-control probe must expose that: length AUC high, geometry
    # AUC no better than length.
    rng = np.random.default_rng(0)
    turns = []
    for i in range(60):
        correct = i % 2 == 0
        m = _metrics(  # geometry = noise, length tracks correctness
            mean_speed=float(rng.normal(3.0, 1.0)),
            n_output_tokens=int(rng.normal(120 if correct else 30, 4)),
        )
        turns.append(LabeledTurn("reasoning", correct, False, m))
    rep = evaluate_floor_test(turns)
    lc = rep.length_control["reasoning"]["correctness"]
    assert lc.length_auc is not None and lc.length_auc > 0.85  # length separates cleanly
    assert lc.geometry_auc is not None
    assert lc.length_auc > lc.geometry_auc  # length beats geometry -> the confound
    assert lc.marginal is not None  # combined - length is computable
    assert "difficulty" in rep.length_control["reasoning"]


def test_length_control_span_aware_length() -> None:
    # The length feature must span the SAME tokens the geometry did: output-only by
    # default, prompt+output for span="full" (else a full-span corpus with a prompt-length
    # confound escapes the control -- the most severe review finding on this diagnostic).
    from pumpkinspice.introspect.evaluate import _length_of

    assert _length_of(_metrics(mean_speed=1.0, n_output_tokens=50)) == 50.0  # output span
    assert (
        _length_of(  # full span: n_prompt_tokens(3) + n_output_tokens(50)
            _metrics(mean_speed=1.0, n_output_tokens=50, trajectory_span="full")
        )
        == 53.0
    )


def test_empty_turns_raises() -> None:
    with pytest.raises(ValueError, match="no turns"):
        evaluate_floor_test([])


def test_incommensurable_corpus_raises_actionably() -> None:
    # mixed d_rho threshold sets -> actionable error, not a bare KeyError deep in numpy
    a = LabeledTurn("t", True, False, _metrics(mean_speed=1.0))
    b = LabeledTurn("t", False, True, _metrics(mean_speed=2.0))
    object.__setattr__(b.metrics, "d_rho", {0.5: 1, 0.9: 2})  # drop the 0.75 threshold
    with pytest.raises(ValueError, match="d_rho thresholds"):
        evaluate_floor_test([a, b])

    # mixed depths -> the rho-curve stack would be ragged; caught up front
    c = LabeledTurn("t", True, False, _metrics(mean_speed=1.0, n_layers=6))
    with pytest.raises(ValueError, match="n_layers"):
        evaluate_floor_test([LabeledTurn("t", True, False, _metrics(mean_speed=1.0)), c])

    # mixed KNOWN replay dtype -> bf16 and fp32 perturb the geometry; refuse to pool them
    fp = LabeledTurn("t", True, False, _metrics(mean_speed=1.0, dtype="float32"))
    bf = LabeledTurn("t", False, True, _metrics(mean_speed=2.0, dtype="bfloat16"))
    with pytest.raises(ValueError, match="replay dtype"):
        evaluate_floor_test([fp, bf])


def test_unknown_dtype_is_unverified_not_blocking() -> None:
    # 'unknown' (pre-provenance) must not be matched by equality: it should neither block
    # extending a float32 corpus nor silently pool -- it warns and is exempt from the check.
    turns = _corpus("tool_use", 20)
    for i, t in enumerate(turns):
        object.__setattr__(t.metrics, "dtype", "float32" if i % 2 else "unknown")
    evaluate_floor_test(turns)  # float32 + unknown: must NOT raise (unknown is unverified)


def test_rho_summary_reports_both_ranges() -> None:
    turns = [
        LabeledTurn("t", True, False, _metrics(mean_speed=1.0, rho_block=[0.9, 0.6, 0.3, 0.0]))
    ]
    rc = evaluate_floor_test(turns).rho_curves
    assert rc is not None
    assert rc.block_range == pytest.approx(0.9)
    assert rc.mlp_range == pytest.approx(0.0)  # rho_mlp is all zeros in the fixture


def test_grouped_cv_deconfounds_task_identity() -> None:
    # feature = one-hot TASK id (pure task identity, no generalizable difficulty signal);
    # label = the task's difficulty class. Ungrouped CV memorizes each task's one-hot dim and
    # predicts the label; grouped CV holds out whole tasks, whose dims were never trained on,
    # so it collapses toward chance -- exactly the Confound-A deconfound the difficulty kill needs.
    from pumpkinspice.introspect.evaluate import _cv_probe_auc

    tasks = ["easyA", "easyB", "hardC", "hardD"]
    task_label = {"easyA": 0, "easyB": 0, "hardC": 1, "hardD": 1}
    x_rows, y_rows, groups = [], [], []
    for i, task in enumerate(tasks):
        for _ in range(8):
            onehot = [0.0, 0.0, 0.0, 0.0]
            onehot[i] = 1.0
            x_rows.append(onehot)
            y_rows.append(task_label[task])
            groups.append(task)
    x = np.array(x_rows)
    y = np.array(y_rows)
    ungrouped = _cv_probe_auc(x, y, seed=0)
    grouped = _cv_probe_auc(x, y, seed=0, groups=groups)
    assert ungrouped is not None and ungrouped > 0.9  # memorizes task identity
    assert grouped is not None and grouped < 0.7  # cannot generalize to held-out tasks


def test_group_roundtrips_and_is_backward_compatible() -> None:
    from pumpkinspice.introspect.evaluate import labeled_turn_from_dict, labeled_turn_to_dict

    lt = LabeledTurn("planning", True, False, _metrics(mean_speed=1.0), group="v2_chicken")
    assert labeled_turn_from_dict(labeled_turn_to_dict(lt)).group == "v2_chicken"
    # a pre-grouped-CV metrics row (no "group" key) loads as ungrouped ""
    d = labeled_turn_to_dict(lt)
    del d["group"]
    assert labeled_turn_from_dict(d).group == ""


def test_report_counts_distinct_task_groups() -> None:
    turns = [
        LabeledTurn("planning", i % 2 == 0, "cow" in g, _metrics(mean_speed=float(i)), group=g)
        for g in ("v2_chicken", "v2_yellow_slime", "v2_cow")
        for i in range(4)
    ]
    report = evaluate_floor_test(turns)
    assert report.n_by_type["planning"] == 12  # episode count
    assert report.n_groups_by_type["planning"] == 3  # effective kill-#1 units (distinct tasks)
