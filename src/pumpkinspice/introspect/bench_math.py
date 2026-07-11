"""MATH reasoning benchmark runner (issue #7's second task type).

Decodes Hendrycks MATH problems through any Decoder plugin into Turn-shaped
captures, so the SAME replay rig and evaluator consume MATH reasoning turns and
HeroBench tool-use turns uniformly. MATH is the locked reasoning corpus because it
ships human-annotated difficulty levels 1-5 -- an INDEPENDENT hard/easy label for
#7's kill #1 -- alongside answer-correctness for kill #2.

Load MATH from a local directory of the standard release JSON (``{problem, level,
type, solution}`` files); no ``datasets`` dependency and no network, which also
sidesteps the Hub takedown of the original repo. Point ``--data-dir`` at your copy.

Grading pulls the last ``\\boxed{...}`` from the model output and the gold solution,
then checks equivalence: first the canonical MATH string normalization (fractions,
sqrt, units, spacing), then -- via the optional ``math-verify`` grader (``introspect``
extra) -- LaTeX/sympy equivalence that repairs false-negatives the string match misses
(bmatrix vs pmatrix, 1/4 vs 0.25), with a guard against math-verify's set leniency
(over-answers like "-1, 2" vs "2"). Core-only installs fall back to string match alone.

Pure/offline-testable: the loader, prompt, and grader are plain functions; the
runner takes an injected Decoder and Capture, so tests use a fake decoder.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pumpkinspice.contracts import Turn

from ..logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pumpkinspice.plugins.decoder_openai_compat import DecodeResult

log = get_logger("pumpkinspice.mathbench")

# Optional robust grader (sympy/LaTeX equivalence). The string normalization below misses
# mathematically-equal answers (bmatrix vs pmatrix, 1/4 vs 0.25, reordered tuples), which
# show up as false-NEGATIVE grades. math-verify closes most of that gap. Guarded so a
# core-only install still grades (by string match alone); enable via the `introspect` extra.
try:
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _HAS_MATH_VERIFY = True
except ImportError:  # pragma: no cover - exercised by the core-only CI matrix
    _HAS_MATH_VERIFY = False

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


class _BatchDecoder(Protocol):
    # The batched runner fans this out concurrently; it returns the result by value (no shared
    # last_* snapshot), unlike complete(). Satisfied by OpenAICompatDecoder (vLLM/LMStudio).
    def decode_one(self, prompt: str, sampler: dict[str, Any] | None = None) -> DecodeResult: ...


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
        # Check the limit up front so limit=0 yields [] (not one item) and files past
        # the cap are not even parsed.
        if limit is not None and len(problems) >= limit:
            break
        # Name the offending file: a MATH dir is thousands of JSONs, so a bare
        # JSONDecodeError / KeyError from one bad file is near-impossible to locate.
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"{path}: malformed MATH problem JSON: {exc}") from exc
        # A corpus dir may legitimately hold a non-problem JSON (e.g. a manifest/index list);
        # skip anything that is not a problem-shaped object rather than crashing on it. A dict
        # that IS problem-shaped but missing a required key still fails loudly below.
        if not isinstance(data, dict):
            continue
        try:
            problem = str(data["problem"])
            solution = str(data["solution"])
        except KeyError as exc:
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
    return problems


# The paper's chain-of-thought prompting styles (arXiv:2607.01571, Appendix B). The main
# results pool `medium` + `long`; `short` is included for completeness. Varying the style is
# one lever the multi-sample runner uses to spread the trajectory distribution per question.
COT_STYLES: dict[str, str] = {
    "medium": "Think step by step. Show your reasoning, then give the final answer in \\boxed{}.",
    "long": (
        "Overthink this. Consider multiple approaches. Solve step by step, then second-guess "
        "yourself. Check your work using a different method. Ask: what could I be missing? What "
        "if I made an error? Keep thinking until you're fully satisfied. Answer in \\boxed{}."
    ),
    "short": (
        "Go with your instinct. Write only the essential steps - no extra explanation. "
        "Answer in \\boxed{}."
    ),
}


def build_prompt(problem: str, style: str = "default") -> str:
    """Render the user-message prompt for a problem. ``default`` is the repo's own competent
    single-pass prompt; the paper's ``medium``/``long``/``short`` styles (COT_STYLES) prepend
    the exact instruction from arXiv:2607.01571 so the multi-sample runner can reproduce it."""
    if style == "default":
        return (
            "Solve the following math problem. Reason step by step, then give the final "
            "answer enclosed in \\boxed{}.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
    try:
        instruction = COT_STYLES[style]
    except KeyError:
        raise ValueError(
            f"unknown prompt style {style!r}; choose from {sorted(COT_STYLES)} or 'default'"
        ) from None
    return f"{instruction}\n\nProblem: {problem}"


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


_PLAIN_NUMBER_RE = re.compile(r"[-+]?[\d,]+(\.\d+)?")


def _looks_like_plain_number(s: str) -> bool:
    """A bare number, optionally with thousands-separator commas (so its commas are NOT list
    separators): "1,000", "-42", "3.14". Used to exempt such answers from the over-answer guard."""
    return bool(_PLAIN_NUMBER_RE.fullmatch(s.strip()))


def _math_verify_equiv(a: str, b: str) -> bool:
    """LaTeX/sympy equivalence via math-verify (handles notation/value equivalence the string
    normalization misses: bmatrix vs pmatrix, 1/4 vs 0.25). Guarded: False if math-verify is
    absent or cannot parse (a parse failure is just "not equivalent by this path", never a
    raise).

    math-verify is LENIENT on lists/sets -- an OVER-answer ("-1, 2" vs "2", extra term) OR an
    UNDER-answer ("2" vs "-1, 2", a subset of a multi-answer gold) can match, both false grades
    (positive and negative-turned-positive) that inflate correctness. Reject ANY comma-count
    MISMATCH in either direction, EXCEPT when both sides are plain numbers (the commas are then
    thousands separators, not list separators)."""
    if not _HAS_MATH_VERIFY:
        return False
    if a.count(",") != b.count(",") and not (
        _looks_like_plain_number(a) and _looks_like_plain_number(b)
    ):
        return False
    try:
        return bool(_mv_verify(_mv_parse(b), _mv_parse(a)))
    except Exception:
        return False


def is_equiv(a: str | None, b: str | None) -> bool:
    # A missing extraction on either side is never a correct grade (an unparseable answer
    # cannot be scored right by accident -- stricter than canonical MATH). Otherwise: the
    # fast exact-normalized string match, then the robust math-verify equivalence, which
    # fixes false-negatives the string match misses (bmatrix vs pmatrix, 1/4 vs 0.25).
    if a is None or b is None:
        return False
    return normalize_answer(a) == normalize_answer(b) or _math_verify_equiv(a, b)


def grade(model_output: str, gold_solution: str) -> tuple[bool, str | None, str | None]:
    """Return (correct, predicted_answer, gold_answer) from the model output and the
    gold solution, both by their last boxed expression."""
    pred = strip_boxed(last_boxed_string(model_output))
    gold = strip_boxed(last_boxed_string(gold_solution))
    return is_equiv(pred, gold), pred, gold


def regrade_rows(rows: list[dict[str, Any]]) -> int:
    """Recompute each MATH capture row's ``outcome.correct`` from its stored ``predicted`` /
    ``gold`` with the CURRENT grader. Returns the count of rows whose ``correct`` flag flipped.
    Lets a grader improvement relabel captures WITHOUT re-decoding -- the model output (and its
    extracted answers) are fixed; only the equivalence verdict changes."""
    changed = 0
    for r in rows:
        oc = r.get("outcome")
        if not isinstance(oc, dict):
            continue
        new = is_equiv(oc.get("predicted"), oc.get("gold"))
        if bool(oc.get("correct")) != new:
            oc["correct"] = new
            changed += 1
    return changed


# --- runner -----------------------------------------------------------------


def _fan_out(
    n: int,
    decode: Callable[[int], DecodeResult],
    *,
    workers: int,
) -> Iterator[tuple[int, DecodeResult | None, Exception | None]]:
    """Run ``decode(i)`` for i in range(n) CONCURRENTLY, yielding ``(i, result, exc)`` in
    completion order -- ``exc`` is set (and ``result`` None) iff that decode raised. The shared
    executor skeleton for the batched and multi-sample MATH runners so their fan-out cannot
    drift; each caller owns its own per-result handling (grading, drops, progress logging)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(decode, i): i for i in range(n)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                yield i, fut.result(), None
            except Exception as exc:  # caller decides how to log + skip
                yield i, None, exc


def _math_turn(
    index: int,
    p: MathProblem,
    *,
    prompt: str,
    raw: str,
    reasoning: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    request: dict[str, Any],
    model: str,
    model_info: dict[str, Any],
    hard_level: int,
    style: str | None = None,
    sample: int | None = None,
) -> Turn:
    """Grade one decoded MATH answer and build its Turn-shaped capture row. Shared by the
    sequential, batched, and multi-sample runners so a row is identical whichever path produced
    it. ``style``/``sample`` (multi-sample only) record which prompt style and which sample of a
    question this trajectory is -- the trajectory's identity within its question group."""
    correct, pred, gold = grade(raw, p.solution)
    world_state: dict[str, Any] = {
        "task_type": "reasoning",
        "subject": p.subject,
        "level": p.level,
    }
    outcome: dict[str, Any] = {
        "task_type": "reasoning",
        "correct": correct,
        "level": p.level,
        "hard": p.level >= hard_level,
        "subject": p.subject,
        "predicted": pred,
        "gold": gold,
        # "length" = the trace hit the token/context cap before it could emit \boxed{}, so a
        # resulting "incorrect" is a truncation artifact, not a wrong answer (the length
        # confound). Surfaced so the floor-test analysis can exclude/condition on it.
        "truncated": finish_reason == "length",
    }
    if style is not None:
        world_state["style"] = style
        outcome["style"] = style
    if sample is not None:
        world_state["sample"] = sample
        outcome["sample"] = sample
    return Turn(
        index=index,
        task=p.problem_id,
        world_state=world_state,
        retrieval={},
        prompt=prompt,
        raw_output=raw,
        action={},
        outcome=outcome,
        timings_ms={},
        reasoning=reasoning,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # Decode provenance -- the reasoning (MATH) arm must record its IV too, else no-think
        # and baseline math runs are indistinguishable.
        decode=request,
        model_info=model_info,
        finish_reason=finish_reason,
    )


def run_math_benchmark(
    decoder: _Decoder,
    problems: list[MathProblem],
    capture: _Capture,
    *,
    hard_level: int = DEFAULT_HARD_LEVEL,
    sampler: dict[str, Any] | None = None,
) -> list[Turn]:
    """Decode each problem SEQUENTIALLY, grade it, and record a Turn-shaped capture.

    The capture's ``outcome`` carries the labels the evaluator needs: ``correct``
    (answer graded), ``level`` and ``hard`` (independent difficulty), plus the
    extracted ``predicted``/``gold`` for auditing. ``task_type`` is "reasoning".
    See ``run_math_benchmark_batched`` for the concurrent path used on vLLM.
    """
    turns: list[Turn] = []
    # The served precision + context window are run-level, not per-turn; snapshot once and
    # stamp every row so a capture is self-describing without a separate run header.
    minfo = dict(getattr(decoder, "model_info", {}))
    model = str(getattr(decoder, "model", "") or "")
    for i, p in enumerate(problems):
        prompt = build_prompt(p.problem)
        raw = decoder.complete(prompt, sampler=sampler)
        usage = getattr(decoder, "last_usage", {}) or {}
        turn = _math_turn(
            i,
            p,
            prompt=prompt,
            raw=raw,
            reasoning=str(getattr(decoder, "last_reasoning", "") or ""),
            finish_reason=str(getattr(decoder, "last_finish_reason", "") or ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            request=dict(getattr(decoder, "last_request", {})),
            model=model,
            model_info=minfo,
            hard_level=hard_level,
        )
        capture.record(turn)
        turns.append(turn)
    return turns


def run_math_benchmark_batched(
    decoder: _BatchDecoder,
    problems: list[MathProblem],
    capture: _Capture,
    *,
    hard_level: int = DEFAULT_HARD_LEVEL,
    sampler: dict[str, Any] | None = None,
    max_concurrency: int = 8,
    log_every: int = 10,
) -> list[Turn]:
    """Decode all problems CONCURRENTLY (vLLM batches the in-flight requests server-side), so a
    single-pass MATH set finishes in a fraction of the sequential wall-clock. Same graded,
    labeled Turn rows as ``run_math_benchmark``.

    Captures are written AS each decode completes (from the main thread, so the append-only sink
    stays single-writer) -- progressive and crash-safe for a long run. A decode that raises
    (timeout, transient 5xx) is logged and skipped, not fatal; the skipped count is logged at the
    end (no silent loss). ``max_concurrency`` bounds in-flight requests; vLLM queues the rest.
    """
    minfo = dict(getattr(decoder, "model_info", {}))
    model = str(getattr(decoder, "model", "") or "")
    prompts = [build_prompt(p.problem) for p in problems]
    n = len(problems)
    turns_by_index: dict[int, Turn] = {}
    failed = 0
    workers = max(1, min(max_concurrency, n or 1))
    log.info("MATH batched: %d problems, concurrency=%d", n, workers)

    def _decode(i: int) -> DecodeResult:
        return decoder.decode_one(prompts[i], sampler)

    for i, res, exc in _fan_out(n, _decode, workers=workers):
        p = problems[i]
        if exc is not None or res is None:  # one bad decode must not abort the whole run
            failed += 1
            log.warning("MATH decode %d (%s L%d) failed, skipped: %s", i, p.subject, p.level, exc)
            continue
        turn = _math_turn(
            i,
            p,
            prompt=prompts[i],
            raw=res.content,
            reasoning=res.reasoning,
            finish_reason=res.finish_reason,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            request=dict(res.request),
            model=model,
            model_info=minfo,
            hard_level=hard_level,
        )
        capture.record(turn)  # main-thread write -> single-writer, crash-safe, progressive
        turns_by_index[i] = turn
        done = len(turns_by_index)
        if done % log_every == 0 or done + failed == n:
            log.info("MATH batched: %d/%d decoded (%d failed)", done, n, failed)
    if failed:
        log.warning("MATH batched: %d/%d problems failed to decode and were skipped", failed, n)
    # Return in problem order (captures were written in completion order; each row self-indexes).
    return [turns_by_index[i] for i in sorted(turns_by_index)]


def run_math_multisample(
    decoder: _BatchDecoder,
    problems: list[MathProblem],
    capture: _Capture,
    *,
    styles: tuple[str, ...] = ("medium", "long"),
    samples_per_style: int = 5,
    temperature: float = 0.7,
    seed_base: int = 0,
    hard_level: int = DEFAULT_HARD_LEVEL,
    min_tokens: int = 30,
    max_concurrency: int = 8,
    log_every: int = 50,
) -> list[Turn]:
    """Decode N stochastic trajectories PER (question, prompt style) and record a per-trajectory
    Turn. This is the correctness-signal protocol of arXiv:2607.01571: many samples per question
    at a nonzero temperature yield a mix of correct/incorrect traces even from a capable model,
    so correctness has real variance to predict; and each Turn's group is the question
    (``Turn.task = problem_id``), so the evaluator can hold whole questions out.

    Each (question, style, sample) is one job with a DISTINCT seed (reproducible). Sampling
    overrides only temperature + seed; top_k / top_p come from the decoder config (vLLM's
    defaults leave the distribution unmodified), so the paper's T=0.7-only spec is honored.
    Jobs fan out concurrently (vLLM batches them). A trajectory shorter than ``min_tokens``
    completion tokens is dropped as invalid (the paper's floor); a decode that raises is logged
    and skipped. Both counts are logged (no silent loss). Captures are written as each completes
    (crash-safe).
    """
    minfo = dict(getattr(decoder, "model_info", {}))
    model = str(getattr(decoder, "model", "") or "")
    # One job per (problem, style, sample); the enumerate index is the trajectory's unique id.
    jobs: list[tuple[int, str, int, int, str]] = []  # (problem_idx, style, sample, seed, prompt)
    seed = seed_base
    for pi, p in enumerate(problems):
        for style in styles:
            prompt = build_prompt(p.problem, style=style)
            for si in range(samples_per_style):
                jobs.append((pi, style, si, seed, prompt))
                seed += 1
    n = len(jobs)
    turns_by_index: dict[int, Turn] = {}
    failed = dropped = 0
    workers = max(1, min(max_concurrency, n or 1))
    log.info(
        "MATH multisample: %d problems x %d styles x %d samples = %d trajectories, concurrency=%d",
        len(problems),
        len(styles),
        samples_per_style,
        n,
        workers,
    )

    def _decode(k: int) -> DecodeResult:
        # Faithful sampling: override ONLY temperature + seed. top_k / top_p come from the
        # decoder config -- vLLM's greedy defaults are top_k=-1 ("all tokens") and top_p=1.0
        # ("no nucleus"), so with the paper specifying only T this leaves the sampled
        # distribution unmodified and never sends an out-of-range top_k=0.
        _pi, _style, _si, jseed, jprompt = jobs[k]
        return decoder.decode_one(jprompt, {"temperature": temperature, "seed": jseed})

    for k, res, exc in _fan_out(n, _decode, workers=workers):
        pi, style, si, _seed, prompt = jobs[k]
        p = problems[pi]
        if exc is not None or res is None:  # one bad decode must not abort the whole run
            failed += 1
            log.warning(
                "multisample q=%s style=%s sample=%d failed, skipped: %s",
                p.problem_id,
                style,
                si,
                exc,
            )
            continue
        if res.completion_tokens < min_tokens:
            dropped += 1  # too-short trajectory is invalid (paper's floor)
            continue
        turn = _math_turn(
            k,
            p,
            prompt=prompt,
            raw=res.content,
            reasoning=res.reasoning,
            finish_reason=res.finish_reason,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            request=dict(res.request),
            model=model,
            model_info=minfo,
            hard_level=hard_level,
            style=style,
            sample=si,
        )
        capture.record(turn)
        turns_by_index[k] = turn
        kept = len(turns_by_index)
        if kept % log_every == 0 or kept + failed + dropped == n:
            log.info(
                "MATH multisample: %d kept, %d dropped(<%d tok), %d failed of %d",
                kept,
                dropped,
                min_tokens,
                failed,
                n,
            )
    if failed or dropped:
        log.warning(
            "MATH multisample: %d/%d trajectories dropped (short) and %d failed to decode",
            dropped,
            n,
            failed,
        )
    return [turns_by_index[j] for j in sorted(turns_by_index)]
