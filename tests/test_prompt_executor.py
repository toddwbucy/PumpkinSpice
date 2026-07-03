"""Stage-4 plan executor: the harness holds the model's JSON plan, advances steps
mechanically against world state, and hands back one step per turn."""

from __future__ import annotations

import json

from pumpkinspice.contracts import BeliefNode, RetrievalResult, Turn, WorldState
from pumpkinspice.plugins.prompt_executor import ExecutorPromptBuilder, _condition_met

RET = RetrievalResult(
    query="q",
    nodes=[BeliefNode(id="r1", text="copper_dagger: 6x copper at (2,1)", score=0.9)],
    latency_ms=0.0,
)

PLAN = json.dumps(
    {
        "plan": [
            {
                "step": 1,
                "description": "gather 12 copper_ore at (2,0)",
                "done_when": {"inventory": {"copper_ore": 12}},
            },
            {
                "step": 2,
                "description": "level weaponcrafting to 5 by crafting copper_daggers",
                "done_when": {"skill": {"weaponcrafting": 5}},
            },
        ]
    }
)


def _state(**overrides: object) -> WorldState:
    raw: dict[str, object] = {"x": 0, "y": 0, "level": 1, "weaponcrafting_level": 1}
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
        action={"kind": "craft", "args": {}},
        outcome={"ok": ok, "status_code": 200 if ok else 500},
        timings_ms={},
    )


def test_condition_met_forms() -> None:
    inv = {"inventory": [{"code": "copper_ore", "quantity": 12}]}
    assert _condition_met({"inventory": {"copper_ore": 12}}, inv) is True
    assert _condition_met({"inventory": {"copper_ore": 13}}, inv) is False
    assert _condition_met({"skill": {"weaponcrafting": 5}}, {"weaponcrafting_level": 5}) is True
    assert _condition_met({"skill": {"weaponcrafting": 5}}, {"weaponcrafting_level": 4}) is False
    # bare character level via the "level" pseudo-skill
    assert _condition_met({"skill": {"level": 2}}, {"level": 3}) is True
    assert _condition_met({"position": [2, 0]}, {"x": 2, "y": 0}) is True
    assert _condition_met({"position": [2, 0]}, {"x": 1, "y": 0}) is False
    # all listed clauses must hold together
    both = {"inventory": {"copper_ore": 12}, "position": [2, 0]}
    assert _condition_met(both, {**inv, "x": 2, "y": 0}) is True
    assert _condition_met(both, {**inv, "x": 0, "y": 0}) is False
    # no recognized clause -> never auto-advances (garbage keys, empty dict)
    assert _condition_met({}, inv) is False
    assert _condition_met({"vibes": True}, inv) is False


def test_plan_turn_then_execute_turn() -> None:
    pb = ExecutorPromptBuilder({})
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[])
    assert '"plan"' in out and "FIRST action" in out  # no plan yet -> plan prompt
    pb.observe(PLAN + '\n{"action": "move", "args": {"x": 1, "y": 0}}')
    assert pb.plan.startswith("1. gather 12 copper_ore")
    out2 = pb.build(state=_state(), retrieval=RET, task="t", history=[])
    assert "CURRENT STEP 1 of 2" in out2 and "[NOW]" in out2


def test_mechanical_advance_and_stable_plan_text() -> None:
    pb = ExecutorPromptBuilder({})
    pb.observe(PLAN)
    before = pb.plan
    # world now satisfies step 1's done_when -> build shows step 2 as current
    st = _state(inventory=[{"code": "copper_ore", "quantity": 12}])
    out = pb.build(state=st, retrieval=RET, task="t", history=[])
    assert "CURRENT STEP 2 of 2" in out and "[done]" in out
    # advancing is NOT a replan: the captured plan text must not change
    assert pb.plan == before
    # ...and the retrieval query targets the current step
    assert "weaponcrafting" in pb.query_for(state=st, task="t")


def test_step_done_flag_advances_but_new_plan_wins() -> None:
    pb = ExecutorPromptBuilder({})
    pb.observe(PLAN)
    pb.observe('{"action": "move", "args": {}, "step_done": true}')
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[])
    assert "CURRENT STEP 2 of 2" in out
    # a rewritten plan resets to its own step 1 (step_done in the same output ignored)
    new_plan = json.dumps(
        {"plan": [{"step": 1, "description": "just rest", "done_when": {"position": [9, 9]}}]}
    )
    pb.observe(new_plan + '\n{"action": "rest", "args": {}, "step_done": true}')
    out2 = pb.build(state=_state(), retrieval=RET, task="t", history=[])
    assert "CURRENT STEP 1 of 1" in out2 and "just rest" in out2


def test_stuck_nudge_and_exhausted_replan() -> None:
    pb = ExecutorPromptBuilder({"replan_after": 2})
    pb.observe(PLAN)
    fails = [_turn(False), _turn(False)]
    out = pb.build(state=_state(), retrieval=RET, task="t", history=fails)
    assert "ALL FAILED" in out and "REVISE NOW" in out
    ok_hist = [_turn(False), _turn(True)]  # a success resets the streak
    assert "ALL FAILED" not in pb.build(state=_state(), retrieval=RET, task="t", history=ok_hist)
    # every step satisfied but the run continues -> ask for a NEW plan
    done_state = _state(inventory=[{"code": "copper_ore", "quantity": 12}], weaponcrafting_level=5)
    out2 = pb.build(state=done_state, retrieval=RET, task="t", history=[])
    assert "NEW plan" in out2 and '"plan"' in out2


def test_malformed_plan_ignored() -> None:
    pb = ExecutorPromptBuilder({})
    pb.observe('{"plan": "not a list"}')
    pb.observe('{"plan": [{"no_description": true}]}')
    pb.observe("no json at all")
    assert pb.plan == ""  # still no plan -> next build asks again
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[])
    assert "FIRST action" in out


def test_lost_fight_counts_toward_stuck_streak() -> None:
    """HeroBench returns HTTP 200 for a LOST fight -- the stuck detector must not
    read that as progress."""
    pb = ExecutorPromptBuilder({"replan_after": 2})
    pb.observe(PLAN)
    lost = Turn(
        index=0,
        task="t",
        world_state={"x": 4, "y": -1},
        retrieval={},
        prompt="",
        raw_output="",
        action={"kind": "fight", "args": {}},
        outcome={
            "ok": True,
            "status_code": 200,
            "data": {"fight": {"result": "lose", "xp": 0, "drops": []}},
        },
        timings_ms={},
    )
    out = pb.build(state=_state(), retrieval=RET, task="t", history=[lost, lost])
    assert "ALL FAILED" in out and "REVISE NOW" in out
    won = Turn(
        index=1,
        task="t",
        world_state={"x": 4, "y": -1},
        retrieval={},
        prompt="",
        raw_output="",
        action={"kind": "fight", "args": {}},
        outcome={
            "ok": True,
            "status_code": 200,
            "data": {"fight": {"result": "win", "xp": 12, "drops": []}},
        },
        timings_ms={},
    )
    assert "ALL FAILED" not in pb.build(
        state=_state(), retrieval=RET, task="t", history=[lost, won]
    )


def test_condition_met_generic_state_clause() -> None:
    """The domain-agnostic escape hatch: a non-HeroBench World (Hanoi's
    "solved"/"pegs") can express done_when without teaching this module its
    state shape."""
    assert _condition_met({"state": {"solved": True}}, {"solved": True}) is True
    assert _condition_met({"state": {"solved": True}}, {"solved": False}) is False
    assert (
        _condition_met(
            {"state": {"pegs": {"A": [], "B": [], "C": [2, 1]}}},
            {"pegs": {"A": [], "B": [], "C": [2, 1]}},
        )
        is True
    )
    # combines with other clause types (all listed conditions must hold)
    both = {"inventory": {"copper_ore": 1}, "state": {"solved": True}}
    assert (
        _condition_met(both, {"inventory": [{"code": "copper_ore", "quantity": 1}], "solved": True})
        is True
    )
    assert (
        _condition_met(
            both, {"inventory": [{"code": "copper_ore", "quantity": 1}], "solved": False}
        )
        is False
    )
