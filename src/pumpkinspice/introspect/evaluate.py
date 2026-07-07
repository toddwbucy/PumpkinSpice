"""Floor-test evaluator (issues #7, #8): labeled trajectory metrics -> the
pre-registered AUCs, rho-curve diagnostics, and keep/kill verdicts.

This is the scientific payoff of the floor test. It consumes ``LabeledTurn``s (a
task type, a correctness label, a hard/easy difficulty label, and the per-turn
``TrajectoryMetrics`` produced by the replay driver) and reports, in TWO SEPARATE
buckets (per issue #8's instruction to keep #7 and #8 verdicts apart):

#7 (trajectory geometry) -- three pre-registered AUC kill conditions:
  1. d_rho hard/easy separation on the AGENTIC (tool-use) turns; AUC < 0.7 -> dead.
  2. early-kinematics correctness AUC within each task type; AUC < 0.7 -> dead.
  3. cross-task-type transfer of the kinematics probe; AUC < 0.7 -> dead. Note the
     threshold check fires on BOTH collapse toward chance (0.5) AND inversion (AUC
     near 0) -- the conservative reading: a probe that does not transfer, in either
     direction, is not a transferring probe. (Flagged for the operationalization
     sign-off, since #7's wording only names the collapse-to-chance case.)

#8 (per-layer novelty) -- the rho_block / rho_MLP curves aggregated across turns,
with a range/spread statistic and a "flat" flag; the kill ("flat or non-monotone
noise across depth") is a judgment on those curves, reported not auto-decided.

Locked operationalization (the pre-registration choice PR 1/2 deferred to the call
site). The "seven kinematics" collapse into a probe as SEVEN SCALARS: the euclidean
norm of each vector kinematic (mean_position, initial_state, final_state,
mean_velocity) plus the three scalar kinematics (positional_dispersion, mean_speed,
speed_dispersion). This keeps the probe low-dimensional so it cannot overfit a few
hundred turns on a d-thousand feature vector. The classifier is a standardized,
L2-regularized logistic regression scored by pooled out-of-fold cross-validated AUC.
d_rho separation uses a probe over the three d_rho thresholds (per-threshold single
AUCs are also reported as diagnostics).

Pure analysis: numpy always, scikit-learn (the ``evaluate`` extra) lazily for the
probe. Never touches the decoder, a model, or the runtime loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pumpkinspice.introspect.geometry import Array, EarlyKinematics, roc_auc
from pumpkinspice.introspect.replay import TrajectoryMetrics

KINEMATIC_FEATURE_NAMES = (
    "mean_position_norm",
    "positional_dispersion",
    "initial_state_norm",
    "final_state_norm",
    "mean_velocity_norm",
    "mean_speed",
    "speed_dispersion",
)

DEFAULT_KILL_THRESHOLD = 0.7
DEFAULT_AGENTIC_TYPE = "tool_use"
# Below this depth-range in the mean rho_block curve, #8's instrument reads as "flat".
DEFAULT_FLAT_RANGE = 0.05


@dataclass(frozen=True)
class LabeledTurn:
    """One replayed turn plus the independent labels the AUCs are scored against."""

    task_type: str  # e.g. "tool_use" (HeroBench) or "reasoning" (MATH)
    correct: bool  # independent outcome label (kill #2/#3)
    hard: bool  # independent difficulty label; True=hard (kill #1)
    metrics: TrajectoryMetrics


def kinematic_features(m: TrajectoryMetrics) -> Array:
    """The seven kinematics as seven scalars (see the module docstring)."""
    k = m.kinematics
    return np.array(
        [
            float(np.linalg.norm(k.mean_position)),
            k.positional_dispersion,
            float(np.linalg.norm(k.initial_state)),
            float(np.linalg.norm(k.final_state)),
            float(np.linalg.norm(k.mean_velocity)),
            k.mean_speed,
            k.speed_dispersion,
        ],
        dtype=np.float64,
    )


def drho_features(m: TrajectoryMetrics) -> Array:
    """The d_rho counts across thresholds (ascending rho) as a feature vector."""
    return np.array([m.d_rho[rho] for rho in sorted(m.d_rho)], dtype=np.float64)


# --- probes -----------------------------------------------------------------


def _cv_probe_auc(
    features: Array, labels: NDArray[np.int_], *, folds: int = 5, seed: int = 0, c: float = 1.0
) -> float | None:
    """Pooled out-of-fold cross-validated AUC of an L2 logistic probe.

    Returns None when the AUC is undefined: only one class present, or too few
    per-class samples to make >= 2 stratified folds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=int)
    if np.unique(y).size < 2:
        return None
    n_splits = min(folds, int(np.bincount(y).min()))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    for train_idx, test_idx in skf.split(features, y):
        scaler = StandardScaler().fit(features[train_idx])
        clf = LogisticRegression(C=c, max_iter=1000).fit(
            scaler.transform(features[train_idx]), y[train_idx]
        )
        oof[test_idx] = clf.predict_proba(scaler.transform(features[test_idx]))[:, 1]
    return roc_auc(oof, y)


def _transfer_auc(
    train_x: Array,
    train_y: NDArray[np.int_],
    test_x: Array,
    test_y: NDArray[np.int_],
    *,
    c: float = 1.0,
) -> float | None:
    """AUC of a probe trained on one task type and evaluated on another (kill #3).
    None if either split is single-class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ytr = np.asarray(train_y, dtype=int)
    yte = np.asarray(test_y, dtype=int)
    if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
        return None
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=c, max_iter=1000).fit(scaler.transform(train_x), ytr)
    proba = clf.predict_proba(scaler.transform(test_x))[:, 1]
    return roc_auc(proba, yte)


def _single_feature_auc(scores: list[float], labels: list[bool]) -> float | None:
    """AUC of one raw feature as the score; None if single-class."""
    if len(set(labels)) < 2:
        return None
    return roc_auc(np.asarray(scores, dtype=np.float64), np.asarray(labels))


# --- report -----------------------------------------------------------------


@dataclass(frozen=True)
class KillCheck:
    """One pre-registered kill condition and whether it survived."""

    name: str
    auc: float | None  # None = undefined (insufficient class balance)
    threshold: float
    passed: bool | None  # None when auc is None
    note: str = ""


@dataclass(frozen=True)
class RhoCurveSummary:
    """#8 diagnostic: mean per-layer novelty curves and a flatness read."""

    rho_block_mean: list[float]
    rho_mlp_mean: list[float]
    block_range: float  # max - min of the mean block curve
    block_std: float
    mlp_range: float  # max - min of the mean MLP curve (#8's kill reads both curves)
    flat: bool  # block_range < flat_range -> reads as structureless


@dataclass(frozen=True)
class FloorTestReport:
    n_by_type: dict[str, int]
    n_layers: int | None
    # #7 diagnostics
    drho_hard_easy: dict[str, float | None]  # task_type -> probe AUC over d_rho features
    drho_per_threshold: dict[str, dict[float, float | None]]
    kinematics_correctness: dict[str, float | None]  # task_type -> CV probe AUC
    cross_transfer: dict[str, float | None]  # "A->B" -> AUC
    # verdict buckets
    kills_hash7: list[KillCheck]
    rho_curves: RhoCurveSummary | None  # #8 bucket

    def summary(self) -> str:
        lines = ["Floor-test report (issues #7, #8)", "=" * 40]
        lines.append("N by task type: " + ", ".join(f"{k}={v}" for k, v in self.n_by_type.items()))
        lines.append(f"layers: {self.n_layers}")
        lines.append("")
        lines.append("#7 kill conditions:")
        for kc in self.kills_hash7:
            verdict = "UNDEFINED" if kc.passed is None else ("PASS" if kc.passed else "KILL")
            auc = "n/a" if kc.auc is None else f"{kc.auc:.3f}"
            lines.append(
                f"  [{verdict}] {kc.name}: AUC={auc} (>= {kc.threshold}) {kc.note}".rstrip()
            )
        lines.append("")
        lines.append("#8 rho curves (separate bucket):")
        if self.rho_curves is None:
            lines.append("  (no per-layer data)")
        else:
            rc = self.rho_curves
            flag = "FLAT (suspect)" if rc.flat else "structured"
            lines.append(f"  block range={rc.block_range:.3f} std={rc.block_std:.3f} -> {flag}")
        return "\n".join(lines)


def _by_type(turns: list[LabeledTurn]) -> dict[str, list[LabeledTurn]]:
    out: dict[str, list[LabeledTurn]] = {}
    for t in turns:
        out.setdefault(t.task_type, []).append(t)
    return out


def _validate_corpus(turns: list[LabeledTurn]) -> None:
    """Fail fast with an actionable message if the turns are not commensurable.

    The labeled-metrics JSONL may be a concatenation of separate runs (PR 3b). All
    turns must share ONE d_rho threshold set (else the probe columns misalign) and
    ONE n_layers (else the rho-curve stack is ragged), so check both up front rather
    than letting a bare KeyError or ragged-array ValueError surface deep in numpy.
    """
    thresholds = set(turns[0].metrics.d_rho)
    n_layers = turns[0].metrics.n_layers
    dtype = turns[0].metrics.dtype
    for i, t in enumerate(turns):
        if set(t.metrics.d_rho) != thresholds:
            raise ValueError(
                f"turn {i}: d_rho thresholds {sorted(t.metrics.d_rho)} != "
                f"{sorted(thresholds)} (all turns must share one threshold set)"
            )
        if t.metrics.n_layers != n_layers:
            raise ValueError(
                f"turn {i}: n_layers {t.metrics.n_layers} != {n_layers} "
                "(cannot pool rho curves across models of different depth)"
            )
        if t.metrics.dtype != dtype:
            raise ValueError(
                f"turn {i}: replay dtype {t.metrics.dtype!r} != {dtype!r} "
                "(bf16 and fp32 perturb the geometry; do not pool across precisions)"
            )


def evaluate_floor_test(
    turns: list[LabeledTurn],
    *,
    agentic_type: str = DEFAULT_AGENTIC_TYPE,
    threshold: float = DEFAULT_KILL_THRESHOLD,
    flat_range: float = DEFAULT_FLAT_RANGE,
    seed: int = 0,
) -> FloorTestReport:
    """Compute the #7 AUC kills and the #8 rho-curve diagnostic from labeled turns."""
    if not turns:
        raise ValueError("no turns to evaluate")
    _validate_corpus(turns)
    grouped = _by_type(turns)
    n_by_type = {ty: len(v) for ty, v in grouped.items()}
    thresholds = sorted(turns[0].metrics.d_rho)

    drho_probe: dict[str, float | None] = {}
    drho_pt: dict[str, dict[float, float | None]] = {}
    kin_probe: dict[str, float | None] = {}
    for ty, group in grouped.items():
        hard = [t.hard for t in group]
        correct = [t.correct for t in group]
        drho_x = np.array([drho_features(t.metrics) for t in group])
        drho_probe[ty] = _cv_probe_auc(drho_x, np.asarray(hard, dtype=int), seed=seed)
        drho_pt[ty] = {
            rho: _single_feature_auc([float(t.metrics.d_rho[rho]) for t in group], hard)
            for rho in thresholds
        }
        kin_x = np.array([kinematic_features(t.metrics) for t in group])
        kin_probe[ty] = _cv_probe_auc(kin_x, np.asarray(correct, dtype=int), seed=seed)

    cross: dict[str, float | None] = {}
    types = list(grouped)
    for a in types:
        for b in types:
            if a == b:
                continue
            xa = np.array([kinematic_features(t.metrics) for t in grouped[a]])
            ya = np.asarray([t.correct for t in grouped[a]], dtype=int)
            xb = np.array([kinematic_features(t.metrics) for t in grouped[b]])
            yb = np.asarray([t.correct for t in grouped[b]], dtype=int)
            cross[f"{a}->{b}"] = _transfer_auc(xa, ya, xb, yb)

    kills = _build_kills(agentic_type, threshold, drho_probe, kin_probe, cross)
    rho_curves = _summarize_rho(turns, flat_range)
    n_layers = turns[0].metrics.n_layers

    return FloorTestReport(
        n_by_type=n_by_type,
        n_layers=n_layers,
        drho_hard_easy=drho_probe,
        drho_per_threshold=drho_pt,
        kinematics_correctness=kin_probe,
        cross_transfer=cross,
        kills_hash7=kills,
        rho_curves=rho_curves,
    )


def _build_kills(
    agentic_type: str,
    threshold: float,
    drho_probe: dict[str, float | None],
    kin_probe: dict[str, float | None],
    cross: dict[str, float | None],
) -> list[KillCheck]:
    def check(name: str, auc: float | None, note: str = "") -> KillCheck:
        passed = None if auc is None else auc >= threshold
        return KillCheck(name=name, auc=auc, threshold=threshold, passed=passed, note=note)

    kills: list[KillCheck] = []
    # Kill #1 is defined on agentic (non-math) trajectories specifically.
    if agentic_type in drho_probe:
        kills.append(check(f"kill1_drho_hard_easy[{agentic_type}]", drho_probe[agentic_type]))
    else:
        kills.append(
            KillCheck(
                f"kill1_drho_hard_easy[{agentic_type}]",
                None,
                threshold,
                None,
                note=f"no '{agentic_type}' turns",
            )
        )
    # Kill #2: within each task type.
    for ty, auc in kin_probe.items():
        kills.append(check(f"kill2_kinematics_correctness[{ty}]", auc))
    # Kill #3: each cross-type transfer direction.
    for direction, auc in cross.items():
        kills.append(check(f"kill3_transfer[{direction}]", auc))
    return kills


def _summarize_rho(turns: list[LabeledTurn], flat_range: float) -> RhoCurveSummary | None:
    block = np.array([t.metrics.rho_block for t in turns])
    mlp = np.array([t.metrics.rho_mlp for t in turns])
    if block.size == 0 or block.shape[1] == 0:
        return None
    block_mean = block.mean(axis=0)
    mlp_mean = mlp.mean(axis=0)
    block_range = float(block_mean.max() - block_mean.min())
    return RhoCurveSummary(
        rho_block_mean=block_mean.tolist(),
        rho_mlp_mean=mlp_mean.tolist(),
        block_range=block_range,
        block_std=float(block_mean.std()),
        mlp_range=float(mlp_mean.max() - mlp_mean.min()),
        flat=block_range < flat_range,
    )


# --- (de)serialization for the labeled-metrics JSONL the CLI consumes -------


def metrics_to_dict(m: TrajectoryMetrics) -> dict[str, Any]:
    k = m.kinematics
    return {
        "d_rho": {str(rho): int(v) for rho, v in m.d_rho.items()},
        "kinematics": {
            "mean_position": k.mean_position.tolist(),
            "positional_dispersion": k.positional_dispersion,
            "initial_state": k.initial_state.tolist(),
            "final_state": k.final_state.tolist(),
            "mean_velocity": k.mean_velocity.tolist(),
            "mean_speed": k.mean_speed,
            "speed_dispersion": k.speed_dispersion,
            "n_points": k.n_points,
        },
        "rho_block": m.rho_block.tolist(),
        "rho_mlp": m.rho_mlp.tolist(),
        "n_prompt_tokens": m.n_prompt_tokens,
        "n_output_tokens": m.n_output_tokens,
        "n_layers": m.n_layers,
        "dtype": m.dtype,
    }


def metrics_from_dict(d: dict[str, Any]) -> TrajectoryMetrics:
    kd = d["kinematics"]
    kin = EarlyKinematics(
        mean_position=np.asarray(kd["mean_position"], dtype=np.float64),
        positional_dispersion=float(kd["positional_dispersion"]),
        initial_state=np.asarray(kd["initial_state"], dtype=np.float64),
        final_state=np.asarray(kd["final_state"], dtype=np.float64),
        mean_velocity=np.asarray(kd["mean_velocity"], dtype=np.float64),
        mean_speed=float(kd["mean_speed"]),
        speed_dispersion=float(kd["speed_dispersion"]),
        n_points=int(kd["n_points"]),
    )
    return TrajectoryMetrics(
        d_rho={float(rho): int(v) for rho, v in d["d_rho"].items()},
        kinematics=kin,
        rho_block=np.asarray(d["rho_block"], dtype=np.float64),
        rho_mlp=np.asarray(d["rho_mlp"], dtype=np.float64),
        n_prompt_tokens=int(d["n_prompt_tokens"]),
        n_output_tokens=int(d["n_output_tokens"]),
        n_layers=int(d["n_layers"]),
        dtype=str(d.get("dtype", "unknown")),
    )


def labeled_turn_to_dict(t: LabeledTurn) -> dict[str, Any]:
    return {
        "task_type": t.task_type,
        "correct": t.correct,
        "hard": t.hard,
        "metrics": metrics_to_dict(t.metrics),
    }


def labeled_turn_from_dict(d: dict[str, Any]) -> LabeledTurn:
    return LabeledTurn(
        task_type=str(d["task_type"]),
        correct=bool(d["correct"]),
        hard=bool(d["hard"]),
        metrics=metrics_from_dict(d["metrics"]),
    )


def load_labeled_turns(path: str | Path) -> list[LabeledTurn]:
    """Read a JSONL of labeled-metrics rows (one ``labeled_turn_to_dict`` per line)."""
    turns: list[LabeledTurn] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            turns.append(labeled_turn_from_dict(json.loads(line)))
    return turns


def report_to_dict(r: FloorTestReport) -> dict[str, Any]:
    return {
        "n_by_type": r.n_by_type,
        "n_layers": r.n_layers,
        "drho_hard_easy": r.drho_hard_easy,
        "drho_per_threshold": {
            ty: {str(k): v for k, v in d.items()} for ty, d in r.drho_per_threshold.items()
        },
        "kinematics_correctness": r.kinematics_correctness,
        "cross_transfer": r.cross_transfer,
        "kills_hash7": [
            {
                "name": k.name,
                "auc": k.auc,
                "threshold": k.threshold,
                "passed": k.passed,
                "note": k.note,
            }
            for k in r.kills_hash7
        ],
        "rho_curves": None
        if r.rho_curves is None
        else {
            "rho_block_mean": r.rho_curves.rho_block_mean,
            "rho_mlp_mean": r.rho_curves.rho_mlp_mean,
            "block_range": r.rho_curves.block_range,
            "block_std": r.rho_curves.block_std,
            "mlp_range": r.rho_curves.mlp_range,
            "flat": r.rho_curves.flat,
        },
    }
