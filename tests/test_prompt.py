"""The default prompt builder renders in-context turn history."""

from __future__ import annotations

from pumpkinspice.contracts import RetrievalResult, Turn, WorldState
from pumpkinspice.plugins.prompt_default import DefaultPromptBuilder


def _turn(index: int, *, ok: bool, action: dict) -> Turn:
    return Turn(
        index=index,
        task="t",
        world_state={"x": 1, "y": 1, "level": 1},
        retrieval={},
        prompt="",
        raw_output="",
        action=action,
        outcome={"ok": ok, "status_code": 200 if ok else 489, "error": None if ok else "HTTP 489"},
        timings_ms={},
    )


def test_build_renders_history_and_failures() -> None:
    pb = DefaultPromptBuilder({})
    state = WorldState(raw={"x": 2, "y": 2, "level": 1})
    ret = RetrievalResult(query="q", nodes=[], latency_ms=0.0)
    history = [
        _turn(0, ok=False, action={"kind": "move", "args": {"x": 9, "y": 9}}),
        _turn(1, ok=True, action={"kind": "move", "args": {"x": 1, "y": 0}}),
    ]
    out = pb.build(state=state, retrieval=ret, task="t", history=history)
    assert "Recent actions" in out
    assert "turn 0" in out and "turn 1" in out
    assert "FAILED (489" in out  # the failed move is surfaced
    # Reflexion + ReAct scaffold is present (the CoT method)
    assert "Reflect" in out and "Thought" in out and "Action" in out


def test_build_empty_history() -> None:
    pb = DefaultPromptBuilder({})
    state = WorldState(raw={"x": 0, "y": 0})
    ret = RetrievalResult(query="q", nodes=[], latency_ms=0.0)
    out = pb.build(state=state, retrieval=ret, task="t", history=[])
    assert "(no actions yet)" in out


def _turn_with_data(index: int, action: dict, data: dict) -> Turn:
    return Turn(
        index=index,
        task="t",
        world_state={"x": 4, "y": -1, "level": 1},
        retrieval={},
        prompt="",
        raw_output="",
        action=action,
        outcome={"ok": True, "status_code": 200, "error": None, "data": data},
        timings_ms={},
    )


def test_history_surfaces_world_feedback() -> None:
    """The agent's own action feedback (fight result, xp, drops) must reach the
    prompt: a LOST fight is HTTP 200 and would otherwise render as a bare "ok",
    and craft xp is the only way to discover that crafting levels the skill."""
    pb = DefaultPromptBuilder({})
    state = WorldState(raw={"x": 0, "y": 0, "level": 1})
    ret = RetrievalResult(query="q", nodes=[], latency_ms=0.0)
    history = [
        _turn_with_data(
            0,
            {"kind": "fight", "args": {}},
            {"fight": {"result": "lose", "monster": "yellow_slime", "xp": 0, "drops": []}},
        ),
        _turn_with_data(
            1,
            {"kind": "craft", "args": {"code": "copper_dagger", "quantity": 1}},
            {
                "details": {
                    "skill": "weaponcrafting",
                    "xp": 150,
                    "items": [{"code": "copper_dagger", "quantity": 1}],
                }
            },
        ),
    ]
    out = pb.build(state=state, retrieval=ret, task="t", history=history)
    assert "fight LOSE vs yellow_slime" in out and "drops: nothing" in out
    assert "+150 weaponcrafting xp" in out and "1x copper_dagger" in out
