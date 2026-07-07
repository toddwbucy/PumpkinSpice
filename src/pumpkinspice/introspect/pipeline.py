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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pumpkinspice.introspect.evaluate import LabeledTurn, labeled_turn_to_dict
from pumpkinspice.introspect.replay import ReplayModel

# (task_type, correct, hard) extracted from one capture row.
LabelFn = Callable[[dict[str, Any]], tuple[str, bool, bool]]
# The full generated text to teacher-force from one capture row.
OutputFn = Callable[[dict[str, Any]], str]


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
) -> tuple[int, int]:
    """Replay every capture and write labeled-metrics rows. Returns (written,
    skipped). A turn is skipped (not fatal) when its forced continuation is too
    short for a trajectory (< 2 tokens) -- e.g. an empty-output turn."""
    # Read (and parse) the input fully BEFORE truncating the output, so a missing or
    # corrupt captures path does not destroy a previous good metrics file.
    turns = _read_jsonl(captures_path)
    written = 0
    skipped = 0
    with Path(out_path).open("w", encoding="utf-8") as w:
        for turn in turns:
            output = output_fn(turn)
            try:
                metrics = model.replay(str(turn.get("prompt", "")), output)
            except ValueError:
                skipped += 1
                continue
            task_type, correct, hard = label_fn(turn)
            row = labeled_turn_to_dict(LabeledTurn(task_type, correct, hard, metrics))
            w.write(json.dumps(row) + "\n")
            written += 1
    return written, skipped
