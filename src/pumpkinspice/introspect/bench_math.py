"""MATH reasoning benchmark runner (issue #7's second task type).

Decodes Hendrycks MATH problems through any Decoder plugin into Turn-shaped
captures, so the SAME replay rig and evaluator consume MATH reasoning turns and
HeroBench tool-use turns uniformly. MATH is the locked reasoning corpus because it
ships human-annotated difficulty levels 1-5 -- an INDEPENDENT hard/easy label for
#7's kill #1 -- alongside answer-correctness for kill #2.

Load MATH from a local directory of the standard release JSON (``{problem, level,
type, solution}`` files); no ``datasets`` dependency and no network, which also
sidesteps the Hub takedown of the original repo. Point ``--data-dir`` at your copy.

Grading is the canonical MATH approach: pull the last ``\\boxed{...}`` from the
model output and from the gold solution, normalize both (fractions, sqrt, units,
spacing), and compare -- string normalization, not sympy, matching lm-eval/Hendrycks.

Pure/offline-testable: the loader, prompt, and grader are plain functions; the
runner takes an injected Decoder and Capture, so tests use a fake decoder.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pumpkinspice.contracts import Turn

# Level at/above which a MATH problem counts as "hard" for #7's difficulty label.
DEFAULT_HARD_LEVEL = 4


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    problem: str
    solution: str
    level: int  # 1..5, or 0 if unparseable
    subject: str


class _Decoder(Protocol):
    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str: ...


class _Capture(Protocol):
    def record(self, turn: Turn) -> None: ...


_LEVEL_RE = re.compile(r"Level\s*(\d+)")


def load_math_dir(
    root: str | Path,
    *,
    levels: set[int] | None = None,
    subjects: set[str] | None = None,
    limit: int | None = None,
) -> list[MathProblem]:
    """Load MATH problems from a directory tree of release JSON files.

    Walks ``root`` for ``*.json``, each ``{"problem", "level": "Level N", "type",
    "solution"}``. Filters by ``levels`` / ``subjects`` if given; ``limit`` caps the
    count. Sorted by path for reproducibility.
    """
    problems: list[MathProblem] = []
    for path in sorted(Path(root).rglob("*.json")):
        # Name the offending file: a MATH dir is thousands of JSONs, so a bare
        # JSONDecodeError / KeyError from one bad file is near-impossible to locate.
        try:
            data = json.loads(path.read_text())
            problem = str(data["problem"])
            solution = str(data["solution"])
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise ValueError(f"{path}: malformed MATH problem JSON: {exc}") from exc
        m = _LEVEL_RE.search(str(data.get("level", "")))
        level = int(m.group(1)) if m else 0
        subject = str(data.get("type", "")) or path.parent.name
        if levels is not None and level not in levels:
            continue
        if subjects is not None and subject not in subjects:
            continue
        problems.append(
            MathProblem(
                problem_id=str(path.relative_to(root).with_suffix("")),
                problem=problem,
                solution=solution,
                level=level,
                subject=subject,
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


def build_prompt(problem: str) -> str:
    """A competent, retrieval-free math prompt that pins the answer format."""
    return (
        "Solve the following math problem. Reason step by step, then give the final "
        "answer enclosed in \\boxed{}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


# --- grading (canonical MATH boxed extraction + normalization) --------------


def last_boxed_string(s: str) -> str | None:
    """The last ``\\boxed{...}`` / ``\\fbox{...}`` substring, or None.

    Handles both the brace form ``\\boxed{...}`` (brace-matched) and the rarer
    space form ``\\boxed 5`` (up to the next ``$`` or end), matching the canonical
    MATH ``last_boxed_only_string`` so gold solutions in either form parse.
    """
    for cmd in ("\\boxed", "\\fbox"):
        idx = s.rfind(cmd)
        if idx >= 0:
            break
    else:
        return None
    after = s[idx + len(cmd) :]
    if after[:1] == "{":
        depth = 0
        for j, ch in enumerate(after):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[idx : idx + len(cmd) + j + 1]
        return None  # unbalanced braces
    if after[:1] == " ":
        return cmd + " " + after[1:].split("$")[0]
    return None


def strip_boxed(s: str | None) -> str | None:
    """Inner content of a ``\\boxed{...}`` / ``\\fbox{...}`` wrapper."""
    if s is None:
        return None
    for prefix in ("\\boxed{", "\\fbox{"):
        if s.startswith(prefix) and s.endswith("}"):
            return s[len(prefix) : -1]
    for prefix in ("\\boxed ", "\\fbox "):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return None


def _fix_fracs(s: str) -> str:
    # Normalize \frac12, \frac1{2}, etc. to \frac{1}{2}. Faithful to the MATH eval port.
    parts = s.split("\\frac")
    out = parts[0]
    for tail in parts[1:]:
        if not tail:
            out += "\\frac"
            continue
        if tail[0] == "{":
            out += "\\frac" + tail
            continue
        try:
            a, b = tail[0], tail[1]
        except IndexError:
            return s
        if b != "{":
            rest = tail[2:] if len(tail) > 2 else ""
            out += "\\frac{" + a + "}{" + b + "}" + rest
        else:
            rest = tail[1:]
            out += "\\frac{" + a + "}" + rest
    return out


def _fix_a_slash_b(s: str) -> str:
    # Turn a bare "a/b" of simple integers into \frac{a}{b}.
    if s.count("/") != 1:
        return s
    a, b = s.split("/")
    try:
        return "\\frac{" + str(int(a)) + "}{" + str(int(b)) + "}"
    except ValueError:
        return s


def _fix_sqrt(s: str) -> str:
    # \sqrt3 -> \sqrt{3}
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for tail in parts[1:]:
        if tail and tail[0] != "{":
            out += "\\sqrt{" + tail[0] + "}" + tail[1:]
        else:
            out += "\\sqrt" + tail
    return out


def normalize_answer(s: str) -> str:
    """Canonical MATH answer normalization (a faithful subset of ``_strip_string``)."""
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    # Strip ONLY trailing units, not every \text{...}: canonical _remove_right_units
    # splits on "\text{ " (brace + space, the unit-writing convention like "3\text{ cm}")
    # and keeps the left part. A blanket re.sub would delete textual answers
    # (\text{even}, \text{Evelyn}) -> both sides normalize to "" -> false-positive grade.
    if "\\text{ " in s:
        s = s.split("\\text{ ")[0]
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # Take the RHS of a simple "x = ..." equation only (exactly one "=", short LHS),
    # matching canonical; leaves multi-equality strings (x=y=3) untouched.
    parts = s.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        s = parts[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s


def is_equiv(a: str | None, b: str | None) -> bool:
    # Deliberately stricter than canonical (which treats None==None as equivalent):
    # here a missing extraction on either side is never a correct grade, so an
    # unparseable answer cannot be scored right by accident.
    if a is None or b is None:
        return False
    return normalize_answer(a) == normalize_answer(b)


def grade(model_output: str, gold_solution: str) -> tuple[bool, str | None, str | None]:
    """Return (correct, predicted_answer, gold_answer) from the model output and the
    gold solution, both by their last boxed expression."""
    pred = strip_boxed(last_boxed_string(model_output))
    gold = strip_boxed(last_boxed_string(gold_solution))
    return is_equiv(pred, gold), pred, gold


# --- runner -----------------------------------------------------------------


def run_math_benchmark(
    decoder: _Decoder,
    problems: list[MathProblem],
    capture: _Capture,
    *,
    hard_level: int = DEFAULT_HARD_LEVEL,
    sampler: dict[str, Any] | None = None,
) -> list[Turn]:
    """Decode each problem, grade it, and record a Turn-shaped capture.

    The capture's ``outcome`` carries the labels the evaluator needs: ``correct``
    (answer graded), ``level`` and ``hard`` (independent difficulty), plus the
    extracted ``predicted``/``gold`` for auditing. ``task_type`` is "reasoning".
    """
    turns: list[Turn] = []
    for i, p in enumerate(problems):
        prompt = build_prompt(p.problem)
        raw = decoder.complete(prompt, sampler=sampler)
        correct, pred, gold = grade(raw, p.solution)
        usage = getattr(decoder, "last_usage", {}) or {}
        turn = Turn(
            index=i,
            task=p.problem_id,
            world_state={"task_type": "reasoning", "subject": p.subject, "level": p.level},
            retrieval={},
            prompt=prompt,
            raw_output=raw,
            action={},
            outcome={
                "task_type": "reasoning",
                "correct": correct,
                "level": p.level,
                "hard": p.level >= hard_level,
                "subject": p.subject,
                "predicted": pred,
                "gold": gold,
            },
            timings_ms={},
            reasoning=str(getattr(decoder, "last_reasoning", "") or ""),
            model=str(getattr(decoder, "model", "") or ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
        capture.record(turn)
        turns.append(turn)
    return turns
