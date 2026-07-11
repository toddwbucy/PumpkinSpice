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
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pumpkinspice.introspect.geometry import Array, EarlyKinematics, roc_auc
from pumpkinspice.introspect.replay import (
    UNKNOWN_DTYPE,
    TrajectoryMetrics,
    normalize_dtype,
)

log = logging.getLogger(__name__)

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
    # Grouping key for grouped cross-validation (Confound A): the TASK this episode belongs
    # to (e.g. "v2_yellow_slime"). The difficulty probe (kill #1) groups by this so no task's
    # trajectories leak across CV folds -- the probe must find a difficulty signature that
    # GENERALIZES to held-out tasks, not memorize one. "" = ungrouped (falls back to plain CV).
    group: str = ""
    # Raw difficulty level (MATH 1-5; 0 if N/A). The reasoning-arm difficulty probe uses the
    # EXTREMES (1 vs 5) per arXiv:2607.01571, which the binary `hard` flag alone cannot express
    # (it would lump level 3 with level 1). 0 for the agentic arm, which has no per-turn level.
    level: int = 0


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
    features: Array,
    labels: NDArray[np.int_],
    *,
    folds: int = 5,
    seed: int = 0,
    c: float = 1.0,
    groups: Sequence[str] | None = None,
    leave_one_group_out: bool = False,
) -> float | None:
    """Pooled out-of-fold cross-validated AUC of an L2 logistic probe.

    When ``groups`` has >= 2 distinct values, folds are GROUPED: no group's samples straddle
    train/test, so the probe cannot memorize a task and must generalize to held-out groups (the
    Confound-A deconfound). ``leave_one_group_out`` uses LeaveOneGroupOut (each group held out
    once -- the paper's leave-one-QUESTION-out protocol; the pooled OOF AUC is well defined even
    though each single-group test fold is one class); otherwise StratifiedGroupKFold with the
    fold count bounded by the minority class's distinct-group count. Without groups, plain
    stratified CV.

    Returns None when the AUC is undefined: one class only, or too few per-class units for
    >= 2 folds (LOQO needs >= 2 groups per class so no training fold is single-class).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=int)
    if np.unique(y).size < 2:
        return None

    g = np.asarray(groups) if groups is not None else None
    # LOQO must NOT silently degrade to plain CV: if leave-one-question-out is requested but the
    # corpus is not grouped (< 2 distinct groups), refuse (return None) rather than leak
    # same-question trajectories across folds -- which would inflate the AUC under a label that
    # still claims "leave-one-question-out".
    if leave_one_group_out and (g is None or np.unique(g).size < 2):
        return None
    if g is not None and np.unique(g).size >= 2:
        # each (training) fold needs both classes -> require >= 2 groups per class
        per_class_groups = min(np.unique(g[y == cls]).size for cls in np.unique(y))
        if per_class_groups < 2:
            return None
        if leave_one_group_out:
            from sklearn.model_selection import LeaveOneGroupOut

            split = LeaveOneGroupOut().split(features, y, groups=g)
        else:
            from sklearn.model_selection import StratifiedGroupKFold

            n_splits = min(folds, per_class_groups)
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split = splitter.split(features, y, groups=g)
    else:
        from sklearn.model_selection import StratifiedKFold

        n_splits = min(folds, int(np.bincount(y).min()))
        if n_splits < 2:
            return None
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split = splitter.split(features, y)

    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    for train_idx, test_idx in split:
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


def _fmt_auc(a: float | None, spec: str = ".3f") -> str:
    return "n/a" if a is None else format(a, spec)


def _length_of(m: TrajectoryMetrics) -> float:
    """The generation length the trajectory actually spans, so the length confound is
    measured on the SAME tokens the geometry saw: output tokens for span='output'
    (the default), prompt+output for span='full'."""
    if m.trajectory_span == "full":
        return float(m.n_prompt_tokens + m.n_output_tokens)
    return float(m.n_output_tokens)


def _length_auc(lengths: list[float], labels: list[bool]) -> float | None:
    """Deterministic single-feature AUC of length, sign-folded to >= 0.5. Length is a
    monotone predictor, so the raw feature AUC is the right estimate; a CV logistic on a
    single column can rank-invert across folds and DEFLATE it (inflating the geometry's
    apparent margin). Folding to max(a, 1-a) reports separability magnitude, comparable
    to the geometry probe (which learns its own sign)."""
    a = _single_feature_auc(lengths, labels)
    return None if a is None else max(a, 1.0 - a)


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
class LengthControl:
    """Does the geometry beat generation LENGTH? (the #7 length confound.)

    A reasoning model chooses its own generation length (its thinking budget), and that
    length alone often predicts correctness/difficulty (harder/failing problems get more
    tokens). ``d_rho`` and the kinematics are length-sensitive, so a geometry AUC can be
    largely a length artifact. Three AUCs on the SAME labels (keyed in the report by task
    type then kind, so ``geometry_auc`` is the corresponding probe's AUC). Which difficulty
    entry backs a kill depends on the arm: the ``difficulty`` entry (binary hard, full corpus)
    backs kill1 for the AGENTIC arm; the ``difficulty_1v5`` entry (extreme levels only) backs
    kill1_drho_1v5 for any arm with both extremes (e.g. the reasoning arm):
      * ``geometry_auc``  -- the geometry features (kinematics for correctness, d_rho for
        difficulty),
      * ``length_auc``    -- the span length alone, a deterministic single-feature AUC,
      * ``combined_auc``  -- geometry + length together.
    If ``combined`` is not meaningfully above ``length``, the geometry adds little beyond
    length. ``marginal`` (combined - length) is that gap -- a POINT ESTIMATE of a
    difference of cross-validated AUCs with NO confidence interval: values within
    estimator noise (the report's AUC CI is roughly +/-0.065 at n~400) are inconclusive,
    so do NOT read a keep/kill sign off ``marginal`` alone."""

    geometry_auc: float | None
    length_auc: float | None
    combined_auc: float | None

    @property
    def marginal(self) -> float | None:
        """combined - length: the geometry's added value beyond generation length.
        A noisy point estimate (see the class docstring) -- not a verdict on its own."""
        if self.combined_auc is None or self.length_auc is None:
            return None
        return self.combined_auc - self.length_auc


@dataclass(frozen=True)
class FloorTestReport:
    n_by_type: dict[str, int]
    # Effective independent-unit count for the DIFFICULTY probe: distinct task-groups per
    # task_type (kill #1 generalizes across these, so this -- not the episode count -- is the
    # number that bounds its reliability). 1 means ungrouped/single-task (kill #1 undefined).
    n_groups_by_type: dict[str, int]
    # Distinct question-groups among the difficulty EXTREMES (the levels the 1-vs-5 probe uses) --
    # the effective unit count for kill1_drho_1v5, which drops level 3, so it is <= n_groups_by_type
    # and is the number that bounds the 1-vs-5 AUC's reliability.
    n_1v5_groups_by_type: dict[str, int]
    n_layers: int | None
    # #7 diagnostics
    drho_hard_easy: dict[str, float | None]  # task_type -> probe AUC over d_rho features
    # task_type -> d_rho 1-vs-5 difficulty AUC (paper protocol: extreme levels only,
    # leave-one-question-out). None when a type lacks both extremes (e.g. the agentic arm).
    drho_1v5: dict[str, float | None]
    # per-threshold single-feature AUCs. drho_per_threshold decomposes drho_hard_easy (full
    # corpus, binary hard); drho_1v5_per_threshold decomposes drho_1v5 (extreme subset, 1-vs-5)
    # and is sparse -- present only for types where the 1-vs-5 probe was computed.
    drho_per_threshold: dict[str, dict[float, float | None]]
    drho_1v5_per_threshold: dict[str, dict[float, float | None]]
    kinematics_correctness: dict[str, float | None]  # task_type -> CV probe AUC
    cross_transfer: dict[str, float | None]  # "A->B" -> AUC (geometry transfer)
    # length-confound control (issues #7): task_type -> kind ("correctness"/"difficulty")
    # -> geometry vs length. cross_transfer_length is the kill3 length baseline: the same
    # transfer with length as the ONLY feature, so "A->B" geometry transfer can be read
    # against "A->B" length transfer (task types differ in length, so kill3 is the probe
    # most exposed to a transferring length proxy).
    length_control: dict[str, dict[str, LengthControl]]
    cross_transfer_length: dict[str, float | None]  # "A->B" -> length-only transfer AUC
    # verdict buckets
    kills_hash7: list[KillCheck]
    rho_curves: RhoCurveSummary | None  # #8 bucket

    def summary(self) -> str:
        lines = ["Floor-test report (issues #7, #8)", "=" * 40]
        lines.append("N by task type: " + ", ".join(f"{k}={v}" for k, v in self.n_by_type.items()))
        lines.append(
            "distinct task-groups (kill #1 units): "
            + ", ".join(f"{k}={v}" for k, v in self.n_groups_by_type.items())
        )
        lines.append(f"layers: {self.n_layers}")
        if any(v is not None for v in self.drho_1v5.values()):
            lines.append("")
            lines.append("d_rho 1-vs-5 difficulty (leave-one-question-out):")
            for ty, auc in self.drho_1v5.items():
                if auc is not None:
                    q = self.n_1v5_groups_by_type.get(ty, 0)
                    lines.append(f"  {ty}: AUC={_fmt_auc(auc)} ({q} extreme-level questions)")
        lines.append("")
        lines.append("#7 kill conditions:")
        for kc in self.kills_hash7:
            verdict = "UNDEFINED" if kc.passed is None else ("PASS" if kc.passed else "KILL")
            lines.append(
                f"  [{verdict}] {kc.name}: AUC={_fmt_auc(kc.auc)} (>= {kc.threshold}) "
                f"{kc.note}".rstrip()
            )
        lines.append("")
        lines.append("Length control (geometry vs length; marginal is a noisy point estimate):")
        if not self.length_control:
            lines.append("  (no probes)")
        for ty, kinds in self.length_control.items():
            for kind, lc in kinds.items():
                lines.append(
                    f"  {kind}[{ty}]: geom={_fmt_auc(lc.geometry_auc)} "
                    f"length={_fmt_auc(lc.length_auc)} combined={_fmt_auc(lc.combined_auc)} "
                    f"(geom beyond length: {_fmt_auc(lc.marginal, '+.3f')})"
                )
        if self.cross_transfer_length:
            lines.append("  kill3 length baseline (geometry transfer should beat this):")
            for d, a in self.cross_transfer_length.items():
                geo = self.cross_transfer.get(d)
                lines.append(f"    {d}: geometry={_fmt_auc(geo)} vs length={_fmt_auc(a)}")
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
    # dtype: KNOWN precisions must be uniform (bf16 and fp32 perturb the geometry).
    # "unknown" (pre-provenance) is unverified -- it can't be matched by equality (that
    # both fails-open on two different-precision legacy files AND blocks extending an
    # fp32 corpus with new float32 turns), so it warns instead.
    known = {t.metrics.dtype for t in turns if t.metrics.dtype != UNKNOWN_DTYPE}
    if len(known) > 1:
        raise ValueError(
            f"corpus mixes replay dtypes {sorted(known)} -- bf16 and fp32 perturb the "
            "geometry; do not pool across precisions"
        )
    if any(t.metrics.dtype == UNKNOWN_DTYPE for t in turns):
        log.warning(
            "corpus has turns with no dtype provenance ('unknown'); cannot verify they "
            "share a precision with the rest"
        )


def _length_control(
    feat_x: Array,
    lengths: list[float],
    labels: list[bool],
    y: NDArray[np.int_],
    geom: float | None,
    *,
    groups: Sequence[str] | None,
    seed: int,
    leave_one_group_out: bool = False,
) -> LengthControl:
    """Geometry vs length-only vs geometry+length AUC for one probe. The SINGLE place all three
    length controls (correctness / difficulty / 1-vs-5) are built, so their shape and geom/label
    pairing cannot drift. The combined probe uses the SAME grouping (and LOQO flag) as its
    geometry probe, so length cannot smuggle task identity past the deconfound; length alone uses
    the deterministic single-feature AUC (a CV probe on one column deflates it)."""
    length_x = np.array([[x] for x in lengths])
    return LengthControl(
        geometry_auc=geom,
        length_auc=_length_auc(lengths, labels),
        combined_auc=_cv_probe_auc(
            np.hstack([feat_x, length_x]),
            y,
            seed=seed,
            groups=groups,
            leave_one_group_out=leave_one_group_out,
        ),
    )


def evaluate_floor_test(
    turns: list[LabeledTurn],
    *,
    agentic_type: str = DEFAULT_AGENTIC_TYPE,
    threshold: float = DEFAULT_KILL_THRESHOLD,
    flat_range: float = DEFAULT_FLAT_RANGE,
    seed: int = 0,
    difficulty_levels: tuple[int, int] = (1, 5),
) -> FloorTestReport:
    """Compute the #7 AUC kills and the #8 rho-curve diagnostic from labeled turns.

    ``difficulty_levels`` are the (easy, hard) extremes for the reasoning-arm difficulty probe
    (arXiv:2607.01571 uses 1 vs 5): d_rho separating those two levels, leave-one-question-out.
    """
    if not turns:
        raise ValueError("no turns to evaluate")
    _validate_corpus(turns)
    grouped = _by_type(turns)
    n_by_type = {ty: len(v) for ty, v in grouped.items()}
    n_groups_by_type = {ty: len({t.group for t in v}) for ty, v in grouped.items()}
    thresholds = sorted(turns[0].metrics.d_rho)
    easy_lvl, hard_lvl = difficulty_levels

    drho_probe: dict[str, float | None] = {}
    drho_1v5: dict[str, float | None] = {}
    n_1v5_groups_by_type: dict[str, int] = {}
    drho_pt: dict[str, dict[float, float | None]] = {}
    drho_1v5_pt: dict[str, dict[float, float | None]] = {}  # sparse: only types where 1v5 ran
    kin_probe: dict[str, float | None] = {}
    length_control: dict[str, dict[str, LengthControl]] = {}
    for ty, group in grouped.items():
        hard = [t.hard for t in group]
        correct = [t.correct for t in group]
        yh = np.asarray(hard, dtype=int)
        yc = np.asarray(correct, dtype=int)
        # DIFFICULTY (kill #1) groups by TASK so no task leaks across folds (Confound A);
        # CORRECTNESS (kill #2) is per-episode, so each episode is its own unit (plain CV).
        task_groups = [t.group for t in group]
        lengths = [_length_of(t.metrics) for t in group]
        drho_x = np.array([drho_features(t.metrics) for t in group])
        kin_x = np.array([kinematic_features(t.metrics) for t in group])
        drho_probe[ty] = _cv_probe_auc(drho_x, yh, seed=seed, groups=task_groups)
        kin_probe[ty] = _cv_probe_auc(kin_x, yc, seed=seed)

        # 1-vs-5 difficulty (arXiv:2607.01571): the difficulty EXTREMES only (drop "medium"
        # level 3, which the binary `hard` flag would lump with easy), labeled by the hard
        # extreme, leave-one-QUESTION-out. Needs both extremes with >= 2 questions each (else
        # None -- e.g. the agentic arm has no per-turn level). The per-threshold decomposition
        # and length control below use the SAME extreme subset+label, and only when the probe
        # actually ran (so there is no orphan length control with a length AUC but no geometry).
        extreme = [t for t in group if t.level in (easy_lvl, hard_lvl)]
        n_1v5_groups_by_type[ty] = len({t.group for t in extreme})
        drho_1v5[ty] = None
        ex_lc: LengthControl | None = None
        if len({t.level for t in extreme}) >= 2:
            ex_x = np.array([drho_features(t.metrics) for t in extreme])
            ex_hard = [t.level == hard_lvl for t in extreme]
            ex_y = np.asarray(ex_hard, dtype=int)
            ex_g = [t.group for t in extreme]
            ex_len = [_length_of(t.metrics) for t in extreme]
            auc = _cv_probe_auc(ex_x, ex_y, seed=seed, groups=ex_g, leave_one_group_out=True)
            drho_1v5[ty] = auc
            if auc is not None:
                # per-threshold decomposition of the 1-vs-5 headline (its OWN subset+label)
                drho_1v5_pt[ty] = {
                    rho: _single_feature_auc(
                        [float(t.metrics.d_rho[rho]) for t in extreme], ex_hard
                    )
                    for rho in thresholds
                }
                ex_lc = _length_control(
                    ex_x,
                    ex_len,
                    ex_hard,
                    ex_y,
                    auc,
                    groups=ex_g,
                    seed=seed,
                    leave_one_group_out=True,
                )

        # Per-threshold single-feature AUCs of the FULL-corpus binary-hard difficulty probe --
        # the decomposition OF drho_hard_easy[ty], so the two always describe the same subset+
        # label. The 1-vs-5 headline has its own drho_1v5_per_threshold (set above) so neither
        # can be misread as the other's breakdown.
        drho_pt[ty] = {
            rho: _single_feature_auc([float(t.metrics.d_rho[rho]) for t in group], hard)
            for rho in thresholds
        }

        # Does geometry beat length? Built in one place (_length_control) for every kind.
        length_control[ty] = {
            "correctness": _length_control(
                kin_x, lengths, correct, yc, kin_probe[ty], groups=None, seed=seed
            ),
            "difficulty": _length_control(
                drho_x, lengths, hard, yh, drho_probe[ty], groups=task_groups, seed=seed
            ),
        }
        if ex_lc is not None:
            length_control[ty]["difficulty_1v5"] = ex_lc

    cross: dict[str, float | None] = {}
    cross_length: dict[str, float | None] = {}
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
            # kill3 length baseline: does length alone transfer? (kill3 is the probe most
            # exposed to a transferring length proxy, since task types differ in length.)
            la = np.array([[_length_of(t.metrics)] for t in grouped[a]])
            lb = np.array([[_length_of(t.metrics)] for t in grouped[b]])
            cross_length[f"{a}->{b}"] = _transfer_auc(la, ya, lb, yb)

    kills = _build_kills(agentic_type, threshold, drho_probe, drho_1v5, kin_probe, cross)
    rho_curves = _summarize_rho(turns, flat_range)
    n_layers = turns[0].metrics.n_layers

    return FloorTestReport(
        n_by_type=n_by_type,
        n_groups_by_type=n_groups_by_type,
        n_1v5_groups_by_type=n_1v5_groups_by_type,
        n_layers=n_layers,
        drho_hard_easy=drho_probe,
        drho_1v5=drho_1v5,
        drho_per_threshold=drho_pt,
        drho_1v5_per_threshold=drho_1v5_pt,
        kinematics_correctness=kin_probe,
        cross_transfer=cross,
        length_control=length_control,
        cross_transfer_length=cross_length,
        kills_hash7=kills,
        rho_curves=rho_curves,
    )


def _build_kills(
    agentic_type: str,
    threshold: float,
    drho_probe: dict[str, float | None],
    drho_1v5: dict[str, float | None],
    kin_probe: dict[str, float | None],
    cross: dict[str, float | None],
) -> list[KillCheck]:
    def check(name: str, auc: float | None, note: str = "") -> KillCheck:
        passed = None if auc is None else auc >= threshold
        return KillCheck(name=name, auc=auc, threshold=threshold, passed=passed, note=note)

    kills: list[KillCheck] = []
    # Kill #1 (agentic): d_rho separates hard/easy on the agentic arm (binary tier difficulty).
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
    # Kill #1 (reasoning): d_rho separates the 1-vs-5 difficulty extremes on any arm that has
    # them (the paper's MATH difficulty result, AUC ~0.93). Registered per type with a 1-vs-5 AUC.
    for ty, auc in drho_1v5.items():
        if auc is not None:
            kills.append(check(f"kill1_drho_1v5[{ty}]", auc))
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
        "trajectory_span": m.trajectory_span,
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
        # normalize at read too, so a producer that recorded "torch.bfloat16" verbatim
        # is not seen as a mismatch against a "bfloat16" row from the same replay.
        dtype=normalize_dtype(d.get("dtype", UNKNOWN_DTYPE)),
        trajectory_span=str(d.get("trajectory_span", "output")),
    )


def labeled_turn_to_dict(t: LabeledTurn) -> dict[str, Any]:
    return {
        "task_type": t.task_type,
        "correct": t.correct,
        "hard": t.hard,
        "group": t.group,
        "level": t.level,
        "metrics": metrics_to_dict(t.metrics),
    }


def labeled_turn_from_dict(d: dict[str, Any]) -> LabeledTurn:
    return LabeledTurn(
        task_type=str(d["task_type"]),
        correct=bool(d["correct"]),
        hard=bool(d["hard"]),
        metrics=metrics_from_dict(d["metrics"]),
        group=str(d.get("group", "")),  # absent in pre-grouped-CV metric files -> ungrouped
        level=int(d.get("level", 0)),  # absent in pre-1v5 metric files -> 0 (no level)
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
        "n_groups_by_type": r.n_groups_by_type,
        "n_1v5_groups_by_type": r.n_1v5_groups_by_type,
        "n_layers": r.n_layers,
        "drho_hard_easy": r.drho_hard_easy,
        "drho_1v5": r.drho_1v5,
        "drho_per_threshold": {
            ty: {str(k): v for k, v in d.items()} for ty, d in r.drho_per_threshold.items()
        },
        "drho_1v5_per_threshold": {
            ty: {str(k): v for k, v in d.items()} for ty, d in r.drho_1v5_per_threshold.items()
        },
        "kinematics_correctness": r.kinematics_correctness,
        "cross_transfer": r.cross_transfer,
        "length_control": {
            ty: {
                kind: {
                    "geometry_auc": lc.geometry_auc,
                    "length_auc": lc.length_auc,
                    "combined_auc": lc.combined_auc,
                    "marginal": lc.marginal,
                }
                for kind, lc in kinds.items()
            }
            for ty, kinds in r.length_control.items()
        },
        "cross_transfer_length": r.cross_transfer_length,
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
