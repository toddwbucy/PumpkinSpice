"""Tests for the MATH runner: loading, prompt, and (the fiddly part) grading.

All offline: grading is pure string logic, the loader reads temp JSON, and the
runner takes a fake decoder. No dataset, no model.
"""

from __future__ import annotations

import json

import pytest

from pumpkinspice.contracts import Turn
from pumpkinspice.introspect.bench_math import (
    grade,
    is_equiv,
    last_boxed_string,
    load_math_dir,
    normalize_answer,
    run_math_benchmark,
    strip_boxed,
)


def test_last_boxed_and_strip() -> None:
    s = r"We compute \boxed{42} first, then \boxed{\frac{1}{2}} last."
    assert last_boxed_string(s) == r"\boxed{\frac{1}{2}}"  # the LAST one
    assert strip_boxed(last_boxed_string(s)) == r"\frac{1}{2}"
    assert last_boxed_string("no box here") is None
    assert strip_boxed(None) is None


def test_last_boxed_handles_nested_braces() -> None:
    s = r"answer \boxed{x^{2} + \frac{1}{2}}"
    assert last_boxed_string(s) == r"\boxed{x^{2} + \frac{1}{2}}"


def test_normalize_answer_canonicalizes() -> None:
    assert normalize_answer(r"\frac12") == r"\frac{1}{2}"
    assert normalize_answer("1/2") == r"\frac{1}{2}"
    assert normalize_answer("0.5") == r"\frac{1}{2}"
    assert normalize_answer(r"\sqrt3") == r"\sqrt{3}"
    assert normalize_answer(r"50\%") == "50"
    assert normalize_answer(r"\left(3\right)") == "(3)"
    assert normalize_answer("x = 7") == "7"  # take RHS of an equation


def test_normalization_branches() -> None:
    # boxed space-form, and normalization corner cases that hit the fixup helpers
    assert strip_boxed(r"\boxed 5") == "5"
    assert normalize_answer(r"\frac1{2}") == r"\frac{1}{2}"  # \frac1{2} -> \frac{1}{2}
    assert normalize_answer(r"\frac5") == r"\frac5"  # too short to fix -> unchanged
    assert normalize_answer("x/y") == "x/y"  # non-integer slash left alone
    assert normalize_answer(r"\sqrt{9}") == r"\sqrt{9}"  # already braced sqrt
    assert normalize_answer(r"3\text{ cm}") == "3"  # units stripped


def test_text_answers_not_stripped() -> None:
    # Regression: a blanket \text{} strip made every text answer normalize to "" ->
    # any two text answers graded equal. Only trailing units ("\text{ cm}") strip.
    assert normalize_answer(r"\text{even}") == r"\text{even}"  # kept, not emptied
    assert normalize_answer(r"3\text{ cm}") == "3"  # units still removed
    even = r"The parity is \boxed{\text{even}}."
    assert grade(r"...so \boxed{\text{even}}", even)[0] is True
    assert grade(r"...so \boxed{\text{odd}}", even)[0] is False  # was a false positive


def test_boxed_space_form_end_to_end() -> None:
    assert last_boxed_string(r"the answer is \boxed 5") == r"\boxed 5"
    # a gold solution written in the space form now grades end to end
    assert grade(r"result: \boxed{5}", r"hence \boxed 5")[0] is True


def test_equation_split_is_faithful() -> None:
    assert normalize_answer("x=7") == "7"  # short LHS, single '=' -> RHS
    assert normalize_answer("x=y=3") == "x=y=3"  # multi-equality left untouched


def test_load_math_dir_names_bad_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "Algebra").mkdir()
    (tmp_path / "Algebra" / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"bad\.json"):
        load_math_dir(tmp_path)


def test_load_math_dir_skips_non_problem_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A corpus dir may hold a non-problem JSON (e.g. seed_math500's manifest, a list) --
    # that must be skipped, not crash the loader; a real problem still loads.
    _write_math(tmp_path, "Algebra", "1.json", problem="2+2?", level=2, solution=r"\boxed{4}")
    (tmp_path / "manifest.json").write_text(json.dumps([{"unique_id": "x", "level": 1}]))
    problems = load_math_dir(tmp_path)
    assert [p.problem for p in problems] == ["2+2?"]  # manifest skipped, problem kept
    # ...but a problem-shaped dict missing a required key still fails loudly
    (tmp_path / "Algebra" / "broken.json").write_text(json.dumps({"problem": "x"}))  # no solution
    with pytest.raises(ValueError, match=r"broken\.json"):
        load_math_dir(tmp_path)


def test_is_equiv_forms() -> None:
    assert is_equiv(r"\frac{1}{2}", "1/2")
    assert is_equiv(r"\dfrac{1}{2}", r"\frac{1}{2}")
    assert not is_equiv("3", "4")
    assert not is_equiv(None, "3")


def test_grade_end_to_end() -> None:
    gold = r"After algebra, the answer is \boxed{\frac{3}{4}}."
    right = r"...reasoning... so \boxed{3/4}."
    wrong = r"...reasoning... so \boxed{2/4}."
    ok, pred, g = grade(right, gold)
    assert ok and pred == "3/4" and g == r"\frac{3}{4}"
    assert grade(wrong, gold)[0] is False
    assert grade("no box", gold)[0] is False  # unparseable model output


def _write_math(  # type: ignore[no-untyped-def]
    root, subject: str, name: str, *, problem: str, level: int, solution: str
) -> None:
    d = root / subject
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(
        json.dumps(
            {"problem": problem, "level": f"Level {level}", "type": subject, "solution": solution}
        )
    )


def test_load_math_dir_filters_and_parses(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_math(tmp_path, "Algebra", "1.json", problem="1+1?", level=2, solution=r"\boxed{2}")
    _write_math(tmp_path, "Algebra", "2.json", problem="hard?", level=5, solution=r"\boxed{9}")
    _write_math(tmp_path, "Geometry", "3.json", problem="area?", level=5, solution=r"\boxed{6}")

    everything = load_math_dir(tmp_path)
    assert len(everything) == 3
    assert {p.level for p in everything} == {2, 5}

    hard = load_math_dir(tmp_path, levels={4, 5})
    assert len(hard) == 2 and all(p.level == 5 for p in hard)

    alg = load_math_dir(tmp_path, subjects={"Algebra"})
    assert {p.subject for p in alg} == {"Algebra"}

    assert len(load_math_dir(tmp_path, limit=1)) == 1
    assert load_math_dir(tmp_path, limit=0) == []  # zero means zero, not one


class _FakeDecoder:
    """Answers with a fixed boxed value; records the reasoning/usage the runner reads."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.last_reasoning = "chain of thought"
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 7}
        self.model = "fake"
        self.model_info = {
            "backend": "fake",
            "quantization": "none",
            "served_context_length": 32768,
        }
        self.last_finish_reason = "stop"

    def complete(self, prompt: str, *, sampler=None) -> str:  # type: ignore[no-untyped-def]
        return f"thinking... \\boxed{{{self._answer}}}"


class _MemCapture:
    def __init__(self) -> None:
        self.turns: list[Turn] = []

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)


def test_run_math_benchmark_grades_and_labels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_math(tmp_path, "Algebra", "1.json", problem="2+2?", level=2, solution=r"\boxed{4}")
    _write_math(tmp_path, "Algebra", "2.json", problem="hard", level=5, solution=r"\boxed{99}")
    problems = load_math_dir(tmp_path)

    cap = _MemCapture()
    turns = run_math_benchmark(_FakeDecoder("4"), problems, cap, hard_level=4)

    assert len(turns) == 2 and len(cap.turns) == 2
    by_level = {t.outcome["level"]: t for t in turns}
    # the level-2 problem whose gold is 4 -> the fake's "4" is correct and easy
    assert by_level[2].outcome["correct"] is True
    assert by_level[2].outcome["hard"] is False
    assert by_level[2].outcome["task_type"] == "reasoning"
    # the level-5 problem whose gold is 99 -> "4" is wrong and hard
    assert by_level[5].outcome["correct"] is False
    assert by_level[5].outcome["hard"] is True
    # reasoning + usage are carried through for the replay step
    assert turns[0].reasoning == "chain of thought"
    assert turns[0].completion_tokens == 7
    # precision + served context are stamped on every row (self-describing capture)
    assert turns[0].model_info["served_context_length"] == 32768
    assert turns[0].model_info["quantization"] == "none"
    assert turns[1].model_info == turns[0].model_info
    # finish_reason -> per-turn truncation signal (a complete "stop" trace is not truncated)
    assert turns[0].finish_reason == "stop"
    assert by_level[2].outcome["truncated"] is False


def test_build_prompt_styles() -> None:
    from pumpkinspice.introspect.bench_math import COT_STYLES, build_prompt

    default = build_prompt("2+2?")
    assert "Solve the following" in default and "2+2?" in default  # repo's single-pass prompt
    medium = build_prompt("2+2?", style="medium")
    assert medium.startswith("Think step by step") and "2+2?" in medium
    long = build_prompt("2+2?", style="long")
    assert "Overthink this" in long and "second-guess" in long and "2+2?" in long
    assert set(COT_STYLES) == {"medium", "long", "short"}
    with pytest.raises(ValueError, match="unknown prompt style"):
        build_prompt("x", style="bogus")


def test_run_math_multisample_per_trajectory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import httpx

    from pumpkinspice.introspect.bench_math import run_math_multisample

    _write_math(tmp_path, "Algebra", "1.json", problem="2+2?", level=3, solution=r"\boxed{4}")
    problems = load_math_dir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 4096}]})
        body = json.loads(request.content)
        seed = int(body.get("seed", 0))
        # faithful sampling: only temperature+seed overridden; top_k/top_p come from vLLM's
        # greedy defaults (all tokens / no nucleus) -- never the rejected top_k=0 or a 0.95 nucleus
        assert body.get("temperature") == 0.7
        assert body.get("top_k") == -1 and body.get("top_p") == 1
        ctok = 5 if seed == 0 else 50  # seed 0 -> too-short trajectory (dropped)
        ans = "4" if seed % 2 == 0 else "5"  # even seeds correct, odd wrong -> correctness varies
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": rf"work \boxed{{{ans}}}"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": ctok},
            },
        )

    d = _vllm_mock(handler)
    cap = _MemCapture()
    turns = run_math_multisample(
        d, problems, cap, styles=("medium", "long"), samples_per_style=3, min_tokens=30
    )

    # 2 styles x 3 samples = 6 jobs (seeds 0..5); seed 0 is < 30 tokens -> dropped -> 5 kept
    assert len(turns) == 5 and len(cap.turns) == 5
    # every trajectory groups on the QUESTION (for question-level held-out CV) ...
    assert all(t.task == problems[0].problem_id for t in turns)
    # ... and records which style + sample it is, symmetrically in world_state AND outcome
    assert {t.world_state["style"] for t in turns} <= {"medium", "long"}
    assert all("sample" in t.world_state and "sample" in t.outcome for t in turns)
    assert all("style" in t.world_state and "style" in t.outcome for t in turns)
    # distinct reproducible seeds per trajectory (provenance)
    assert len({t.decode.get("seed") for t in turns}) == 5
    # the whole point: correctness has variance to predict (a mix of right and wrong)
    assert {t.outcome["correct"] for t in turns} == {True, False}


def _vllm_mock(handler):  # type: ignore[no-untyped-def]
    import httpx

    from pumpkinspice.plugins.decoder_vllm import VLLMDecoder

    d = VLLMDecoder({"base_url": "http://x", "model": "m", "quantization": "none"})
    d._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    return d


def test_run_math_benchmark_batched_grades_labels_and_stamps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import httpx

    from pumpkinspice.introspect.bench_math import run_math_benchmark_batched

    _write_math(tmp_path, "Algebra", "1.json", problem="2+2?", level=2, solution=r"\boxed{4}")
    _write_math(tmp_path, "Algebra", "2.json", problem="hard", level=5, solution=r"\boxed{99}")
    problems = load_math_dir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 32768}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": r"so \boxed{4}", "reasoning_content": "cot"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        )

    d = _vllm_mock(handler)
    cap = _MemCapture()
    turns = run_math_benchmark_batched(d, problems, cap, hard_level=4, max_concurrency=4)

    assert len(turns) == 2 and len(cap.turns) == 2
    assert [t.index for t in turns] == [0, 1]  # returned in problem order despite concurrency
    by_level = {t.outcome["level"]: t for t in turns}
    assert by_level[2].outcome["correct"] is True  # gold 4, answer 4
    assert by_level[5].outcome["correct"] is False  # gold 99, answer 4
    assert by_level[2].outcome["truncated"] is False  # finish_reason "stop"
    # provenance + fields stamped the same as the sequential path
    assert turns[0].model_info["served_context_length"] == 32768
    assert turns[0].reasoning == "cot" and turns[0].finish_reason == "stop"
    assert turns[0].completion_tokens == 7
    # the concurrent decodes did not touch the sequential snapshot
    assert d.last_reasoning == "" and d.last_request == {}


def test_run_math_benchmark_batched_skips_failed_decode(tmp_path, caplog) -> None:  # type: ignore[no-untyped-def]
    import httpx

    from pumpkinspice.introspect.bench_math import run_math_benchmark_batched

    _write_math(tmp_path, "Algebra", "1.json", problem="ok", level=3, solution=r"\boxed{4}")
    _write_math(tmp_path, "Algebra", "2.json", problem="boom", level=3, solution=r"\boxed{4}")
    problems = load_math_dir(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 32768}]})
        body = json.loads(request.content)
        if "boom" in body["messages"][0]["content"]:
            return httpx.Response(500, json={"error": "kaboom"})  # one decode fails
        return httpx.Response(
            200, json={"choices": [{"message": {"content": r"\boxed{4}"}, "finish_reason": "stop"}]}
        )

    d = _vllm_mock(handler)
    cap = _MemCapture()
    with caplog.at_level("WARNING"):
        turns = run_math_benchmark_batched(d, problems, cap, max_concurrency=2)

    # the good problem is recorded; the failing one is skipped (logged), not fatal
    assert len(turns) == 1 and turns[0].outcome["correct"] is True
    assert any("failed" in r.message and "skipped" in r.message for r in caplog.records)


def test_run_math_benchmark_flags_truncated_traces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # A trace cut off at the cap (finish_reason "length") emits no \boxed{} -> graded
    # "incorrect", but outcome.truncated marks it so the floor-test analysis can exclude it
    # rather than count it as a genuine wrong answer (the length confound).
    _write_math(tmp_path, "Algebra", "1.json", problem="hard", level=5, solution=r"\boxed{42}")
    problems = load_math_dir(tmp_path)

    dec = _FakeDecoder("42")
    dec.last_finish_reason = "length"  # truncated before the answer

    def _no_answer(prompt: str, *, sampler=None) -> str:  # type: ignore[no-untyped-def]
        return "thinking and thinking but never closes..."

    dec.complete = _no_answer  # type: ignore[method-assign]
    turns = run_math_benchmark(dec, problems, _MemCapture())
    assert turns[0].outcome["correct"] is False
    assert turns[0].outcome["truncated"] is True
    assert turns[0].finish_reason == "length"


def test_is_equiv_math_verify_fixes_notation() -> None:
    pytest.importorskip("math_verify")
    # notation- and value-equivalent answers the string normalization alone marks WRONG
    assert (
        is_equiv(
            r"\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}",
            r"\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}",
        )
        is True
    )
    assert is_equiv("0.25", r"\frac{1}{4}") is True
    # genuinely different answers stay wrong; a missing extraction is never correct
    assert is_equiv("7", "8") is False
    assert is_equiv(None, "8") is False
    # comma-count guard, BOTH directions: math-verify's set leniency would mark an over-answer
    # ("-1, 2" vs "2") OR an under-answer ("2" vs "-1, 2", gave 1 of 2 roots) correct -> both
    # inflate correctness, so both must grade False.
    assert is_equiv("-1, 2", "2") is False  # over-answer
    assert is_equiv(r"30^\circ, 45^\circ, 105^\circ", r"105^\circ") is False
    assert is_equiv("2", "-1, 2") is False  # under-answer (the regression)
    assert is_equiv("3", "1, 2, 3") is False
    # ...but thousands-separator commas in a plain number are exempt (they are not a list)
    assert is_equiv("1,000", "1000") is True


def test_regrade_rows_flips_only_stale_labels() -> None:
    pytest.importorskip("math_verify")
    from pumpkinspice.introspect.bench_math import regrade_rows

    rows = [
        # a false NEGATIVE (equivalent notation graded wrong) -> flips to True
        {
            "outcome": {
                "correct": False,
                "predicted": r"\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}",
                "gold": r"\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}",
            }
        },
        {"outcome": {"correct": True, "predicted": "8", "gold": "8"}},  # already correct
        {"outcome": {"correct": False, "predicted": "7", "gold": "8"}},  # genuinely wrong
        {"world_state": {}},  # non-MATH row (no outcome dict) -> skipped, no crash
    ]
    assert regrade_rows(rows) == 1
    assert rows[0]["outcome"]["correct"] is True
    assert rows[1]["outcome"]["correct"] is True
    assert rows[2]["outcome"]["correct"] is False


def test_math_regrade_command_is_atomic_and_robust(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("math_verify")
    import argparse

    from pumpkinspice.cli import _cmd_math_regrade

    cap = tmp_path / "m.jsonl"
    cap.write_text(
        json.dumps(
            {
                "outcome": {
                    "correct": False,
                    "predicted": r"\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}",
                    "gold": r"\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}",
                }
            }
        )
        + "\n"
        + json.dumps({"outcome": None})  # non-dict outcome -> counters must not crash (#4)
        + "\n"
        + json.dumps({"world_state": {}})  # no outcome at all
        + "\n"
    )
    rc = _cmd_math_regrade(argparse.Namespace(capture=str(cap), out=None))
    assert rc == 0
    out_rows = [json.loads(x) for x in cap.read_text().splitlines() if x.strip()]
    assert len(out_rows) == 3  # all rows written back (atomic swap kept the file intact)
    assert out_rows[0]["outcome"]["correct"] is True  # the genuine bmatrix flip applied
