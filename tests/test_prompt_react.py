"""Externalized-reasoning (ReAct, thinking-off) prompt strategy: a cold first-turn plan
plus bounded later steps, ICL-free."""

from __future__ import annotations

from pumpkinspice.contracts import BeliefNode, RetrievalResult, Turn, WorldState
from pumpkinspice.plugins.prompt_react import ReactPromptBuilder

RET = RetrievalResult(
    query="q",
    nodes=[BeliefNode(id="m1", text="chicken: level 1, no resistance", score=0.9)],
    latency_ms=0.0,
)


def _state(**overrides: object) -> WorldState:
    raw: dict[str, object] = {"x": 0, "y": 0, "level": 1}
    raw.update(overrides)
    return WorldState(raw=raw)


def _turn(ok: bool) -> Turn:
    return Turn(
        index=0,
        task="t",
        world_state={"x": 0, "y": 0},
        retrieval={},
        prompt="",
        raw_output="",
        action={"kind": "fight", "args": {}},
        outcome={"ok": ok, "status_code": 200 if ok else 500},
        timings_ms={},
    )


def test_first_turn_elicits_plan_and_action() -> None:
    pb = ReactPromptBuilder({})
    out = pb.build(state=_state(), retrieval=RET, task="beat a chicken", history=[])
    assert "## Goal" in out and "beat a chicken" in out
    assert '{"action": "<verb>", "args"' in out  # action grammar present
    assert "Plan" in out  # the measured first turn asks for a cold plan
    assert "chicken: level 1" in out  # retrieval notes rendered
    assert "BRIEFLY" in out  # externalized -> bounded reasoning instruction


def test_later_turn_uses_history_and_no_fresh_plan_block() -> None:
    pb = ReactPromptBuilder({})
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[_turn(ok=True)])
    assert "Recent actions" in out
    assert 'brief "Thought"' in out  # later turns want Thought+Action, not a new full plan


def test_failed_last_action_adds_nudge() -> None:
    pb = ReactPromptBuilder({})
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[_turn(ok=False)])
    assert "FAILED" in out


def test_query_for_and_name() -> None:
    pb = ReactPromptBuilder({})
    q = pb.query_for(state=_state(), task="beat a chicken")
    assert isinstance(q, str) and q
    assert pb.name == "react"
