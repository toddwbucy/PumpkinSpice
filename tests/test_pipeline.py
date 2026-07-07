"""Tests for the capture -> labeled-metrics bridge (issues #7, #8).

Uses a 2-layer from-config Llama + a fake tokenizer (no download) to replay a
temp capture JSONL, then loads the result back through the evaluator's reader.
Skips if the ``replay`` extra is absent.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pumpkinspice.contracts import Turn  # noqa: E402
from pumpkinspice.introspect.evaluate import load_labeled_turns  # noqa: E402
from pumpkinspice.introspect.pipeline import (  # noqa: E402
    build_output,
    labels_from_outcome,
    replay_captures,
)
from pumpkinspice.introspect.replay import ReplayModel  # noqa: E402


class _FakeTok:
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [(ord(c) % 31) + 1 for c in text]}


def _tiny_model() -> object:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg)


def _capture_row(**over: object) -> dict[str, object]:
    turn = Turn(
        index=0,
        task="t",
        world_state={"task_type": "reasoning"},
        retrieval={},
        prompt="hello world",
        raw_output="abcdef",
        action={},
        outcome={"task_type": "reasoning", "correct": True, "hard": False},
        timings_ms={},
    )
    row = dataclasses.asdict(turn)
    row.update(over)
    return row


def test_build_output_prepends_reasoning() -> None:
    assert build_output({"reasoning": "cot", "raw_output": "ans"}) == "cotans"
    assert build_output({"raw_output": "ans"}) == "ans"
    assert build_output({}) == ""


def test_labels_from_outcome_math_and_herobench() -> None:
    math = {"outcome": {"task_type": "reasoning", "correct": True, "hard": True}}
    assert labels_from_outcome(math) == ("reasoning", True, True)
    # HeroBench-style: correct falls back to `ok`, hard defaults False, type from world_state
    hero = {"outcome": {"ok": True}, "world_state": {"task_type": "tool_use"}}
    assert labels_from_outcome(hero) == ("tool_use", True, False)


def test_replay_captures_writes_labeled_metrics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    caps = tmp_path / "caps.jsonl"
    rows = [
        _capture_row(),
        _capture_row(outcome={"task_type": "reasoning", "correct": False, "hard": True}),
        _capture_row(raw_output="", reasoning=""),  # empty output -> skipped (span < 2)
    ]
    caps.write_text("\n".join(json.dumps(r) for r in rows))

    model = ReplayModel(_tiny_model(), tokenizer=_FakeTok())
    out = tmp_path / "labeled.jsonl"
    written, skipped = replay_captures(model, caps, out)
    model.close()

    assert (written, skipped) == (2, 1)
    turns = load_labeled_turns(out)
    assert len(turns) == 2
    assert turns[0].task_type == "reasoning" and turns[0].correct is True
    assert turns[1].correct is False and turns[1].hard is True
    # the metrics survived serialization and are usable downstream
    assert turns[0].metrics.n_layers == 2
    assert set(turns[0].metrics.d_rho) == {0.5, 0.75, 0.9}
