"""Bridge: capture JSONL -> labeled trajectory-metrics JSONL (issues #7, #8).

Ties the three pieces together. Reads a per-turn capture (HeroBench tool-use turns
or MATH reasoning turns -- both are Turn-shaped), teacher-forces each through the
replay driver to get its TrajectoryMetrics, attaches the independent labels the
evaluator scores against (task type, correctness, hard/easy), and writes the
labeled-metrics JSONL that ``pumpkinspice floortest`` consumes.

Needs the ``replay`` extra to run (it drives a real model); importing this module
does not (ReplayModel imports torch lazily).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pumpkinspice.introspect.evaluate import LabeledTurn, labeled_turn_to_dict
from pumpkinspice.introspect.replay import ReplayModel

log = logging.getLogger(__name__)

# (task_type, correct, hard) extracted from one capture row.
LabelFn = Callable[[dict[str, Any]], tuple[str, bool, bool]]
# The full generated text to teacher-force from one capture row.
OutputFn = Callable[[dict[str, Any]], str]

# Max allowed drift between the re-derived prompt length and the server-reported
# prompt_tokens before a turn is skipped: a real chat-template mismatch shifts the
# count by far more than a BOS/edge-token quirk, and a wrong prompt length misaligns
# the whole span (lo = n_prompt_tokens) with no error otherwise.
PROMPT_TOKEN_DRIFT_TOLERANCE = 4


def build_output(turn: dict[str, Any]) -> str:
    """The full generated text to force: reasoning (chain-of-thought) + the answer.

    Reasoning models emit their thinking separately (captured in ``reasoning``); the
    trajectory of interest is the whole generation, so prepend it to ``raw_output``.
    The reasoning/answer seam is joined without a separator, so (like the prompt/output
    seam in ReplayModel._encode) the re-tokenized ids may differ slightly there from
    the original generation -- within the same stated tolerance.
    """
    reasoning = turn.get("reasoning") or ""
    raw = turn.get("raw_output") or ""
    return reasoning + raw if reasoning else raw


def labels_from_outcome(turn: dict[str, Any]) -> tuple[str, bool, bool]:
    """Default label extraction. ``task_type`` and ``hard`` come from the capture's
    outcome (MATH sets both; HeroBench sets neither, so they fall back); ``correct``
    is ``outcome.correct`` (MATH) or ``outcome.ok`` (HeroBench)."""
    outcome = turn.get("outcome") or {}
    world = turn.get("world_state") or {}
    task_type = str(outcome.get("task_type") or world.get("task_type") or "unknown")
    correct = bool(outcome.get("correct", outcome.get("ok", False)))
    hard = bool(outcome.get("hard", False))
    return task_type, correct, hard


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def replay_captures(
    model: ReplayModel,
    captures_path: str | Path,
    out_path: str | Path,
    *,
    label_fn: LabelFn = labels_from_outcome,
    output_fn: OutputFn = build_output,
    group: str | None = None,
    group_by: str | None = None,
) -> tuple[int, int]:
    """Replay every capture and write labeled-metrics rows. Returns (written, skipped). A turn
    is skipped (not fatal) when its forced continuation is too short for a trajectory
    (< 2 tokens), or when the re-derived prompt length drifts from the server-reported
    ``prompt_tokens`` (a chat-template mismatch that would misalign the span).

    ``group`` is the grouped-CV key stamped on every row -- the TASK these episodes belong to,
    for the difficulty deconfound. Default: the capture filename's task prefix (v2 captures are
    ``<tier>__ep<N>_seed<S>.jsonl``, so ``v2_yellow_slime__ep003...`` -> ``v2_yellow_slime``; a
    name without ``__`` groups by the whole stem). To evaluate v2, replay each episode capture
    (its per-episode ``label_fn`` scores that episode's outcome) and CONCATENATE the metrics --
    every row already carries its task group, so grouped CV sees all tasks. The per-file
    labeling + concatenation orchestration is v2 run-workflow (step 7).

    ``group_by="task"`` instead stamps each row with ITS OWN turn's ``task`` (the question/
    problem id for the multi-sample MATH arm, where many trajectories share one capture file
    but each belongs to a different question), so grouped CV holds whole QUESTIONS out -- the
    correctness protocol of arXiv:2607.01571. A row missing ``task`` falls back to the file
    group. Default (None / "filename") keeps the file-level group above."""
    # Read (and parse) the input fully BEFORE truncating the output, so a missing or
    # corrupt captures path does not destroy a previous good metrics file.
    turns = _read_jsonl(captures_path)
    file_grp = group if group is not None else Path(captures_path).stem.split("__")[0]
    written = 0
    skipped = 0
    with Path(out_path).open("w", encoding="utf-8") as w:
        for turn in turns:
            output = output_fn(turn)
            # Encode first (cheap tokenization) so the parity check can skip a drifted turn
            # BEFORE the expensive hook-capturing forward pass -- the very case it guards.
            input_ids, n_prompt_tokens = model.encode(str(turn.get("prompt", "")), output)
            # Parity check: the re-derived template must match what the serving stack
            # applied, or the span slice (lo = n_prompt_tokens) is wrong for every metric.
            # Ground truth is the server's prompt_tokens (0 = not reported, e.g. offline);
            # a malformed/legacy value coerces to 0 (skip the check), never aborts the run.
            try:
                server_pt = int(turn.get("prompt_tokens") or 0)
            except (TypeError, ValueError):
                server_pt = 0
            if server_pt and abs(n_prompt_tokens - server_pt) > PROMPT_TOKEN_DRIFT_TOLERANCE:
                log.warning(
                    "prompt-token drift on %r: re-derived %d vs server %d -- chat-template "
                    "mismatch, skipping (span would be misaligned)",
                    turn.get("task"),
                    n_prompt_tokens,
                    server_pt,
                )
                skipped += 1
                continue
            try:
                metrics = model.replay_token_ids(input_ids, n_prompt_tokens)
            except ValueError:  # forced continuation too short for a trajectory (< 2 tokens)
                skipped += 1
                continue
            task_type, correct, hard = label_fn(turn)
            # Per-question grouping (multi-sample MATH): the row's own task id, else the file group.
            grp = str(turn.get("task")) if (group_by == "task" and turn.get("task")) else file_grp
            # Raw difficulty level for the 1-vs-5 difficulty probe (0 when absent, e.g. HeroBench).
            # `outcome` may be a non-dict (a list, or None) -> isinstance guard, then coerce.
            oc = turn.get("outcome")
            try:
                level = int(oc.get("level") or 0) if isinstance(oc, dict) else 0
            except (TypeError, ValueError):
                level = 0
            row = labeled_turn_to_dict(
                LabeledTurn(task_type, correct, hard, metrics, group=grp, level=level)
            )
            w.write(json.dumps(row) + "\n")
            written += 1
    return written, skipped
