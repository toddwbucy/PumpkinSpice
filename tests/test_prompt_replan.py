"""Stage-3 replan strategy: a living plan that revises on surprise (unlike Stage 2,
which commits the first plan and never changes it)."""

from __future__ import annotations

from pumpkinspice.contracts import BeliefNode, RetrievalResult, Turn, WorldState
from pumpkinspice.plugins.prompt_replan import ReplanPromptBuilder

STATE = WorldState(raw={"x": 0, "y": 0, "level": 1})
RET = RetrievalResult(
    query="q",
    nodes=[BeliefNode(id="r1", text="copper_dagger: 6x copper", score=0.9)],
    latency_ms=0.0,
)
PLAN_A = (
    '## Plan\n1. Gather copper_ore.\n2. Smelt copper.\n## Action\n{"action": "move", "args": {}}'
)
PLAN_B = '## Plan\n1. A revised plan: fight first.\n## Action\n{"action": "fight", "args": {}}'


def _failed_turn() -> Turn:
    return Turn(
        index=0,
        task="t",
        world_state={"x": 0, "y": 0},
        retrieval={},
        prompt="",
        raw_output="",
        action={"kind": "craft", "args": {}},
        outcome={"ok": False, "status_code": 500, "error": "HTTP 500"},
        timings_ms={},
    )


def test_replan_revises_unlike_stage2() -> None:
    pb = ReplanPromptBuilder({})
    pb.observe(PLAN_A)
    assert "Gather copper_ore" in pb.plan
    # Stage 3 KEY DIFFERENCE: a later "## Plan" REVISES the committed plan.
    pb.observe(PLAN_B)
    assert "revised plan" in pb.plan and "Gather copper_ore" not in pb.plan


def test_replan_nudges_after_failure() -> None:
    pb = ReplanPromptBuilder({})
    pb.observe(PLAN_A)
    # last action failed -> the prompt should invite a keep-or-revise decision
    out = pb.build(state=STATE, retrieval=RET, task="t", history=[_failed_turn()])
    assert "last action FAILED" in out and "REVISE" in out
    # no failure -> no nudge
    out2 = pb.build(state=STATE, retrieval=RET, task="t", history=[])
    assert "last action FAILED" not in out2
    assert "## Your current plan" in out2  # the living plan is shown back


def test_replan_metric_counts_changes() -> None:
    from pumpkinspice.analyze import analyze_turns

    def turn(i: int, plan: str) -> dict:
        return {
            "index": i,
            "world_state": {"x": 0, "y": 0},
            "retrieval": {},
            "action": {"kind": "move", "args": {}},
            "outcome": {"ok": True},
            "timings_ms": {"decode": 1.0},
            "plan": plan,
        }

    # plan set (initial), unchanged, then changed twice = 2 replans
    turns = [turn(0, "A"), turn(1, "A"), turn(2, "B"), turn(3, "C")]
    assert analyze_turns("r", turns).replans == 2


def test_plan_only_output_still_commits_and_revises() -> None:
    """Same terminator edge case as Stage 2: a bare plan with no trailing action
    or header must commit -- and a bare revision must replace it."""
    pb = ReplanPromptBuilder({})
    pb.observe("## Plan\n1. Gather copper_ore.")
    assert "Gather copper_ore" in pb.plan
    pb.observe("## Plan\n1. Fight first instead.")
    assert "Fight first" in pb.plan and "Gather copper_ore" not in pb.plan
