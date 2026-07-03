"""The Stage-2 planning prompt: commit a plan on turn 0, then execute it."""

from __future__ import annotations

from pumpkinspice.contracts import BeliefNode, RetrievalResult, WorldState
from pumpkinspice.plugins.prompt_plan import PlanningPromptBuilder

STATE = WorldState(raw={"x": 0, "y": 0, "level": 1})
RET = RetrievalResult(
    query="q",
    nodes=[BeliefNode(id="r1", text="copper_dagger: 6x copper at weaponsmith", score=0.9)],
    latency_ms=0.0,
)

# A first-turn output in the asked-for format: a plan, then the first action.
FIRST_OUTPUT = """\
## Plan
1. Go to copper_rocks at (2, 0) and gather 8 copper_ore.
2. Smelt 6 copper at the forge.
3. Craft copper_dagger at the weaponsmith.
## Action
{"action": "move", "args": {"x": 1, "y": 0}}
"""


def test_plan_turn_asks_for_a_plan_then_commits_it() -> None:
    pb = PlanningPromptBuilder({})
    # Turn 0: no committed plan yet -> the prompt asks the model to write one.
    first = pb.build(state=STATE, retrieval=RET, task="Craft a copper dagger.", history=[])
    assert '"## Plan"' in first  # turn 0 asks the model to write a plan
    assert "## Your committed plan" not in first  # no committed-plan section yet
    assert pb.plan == ""

    # The loop feeds the model output back; the plan is parsed and committed.
    pb.observe(FIRST_OUTPUT)
    assert "gather 8 copper_ore" in pb.plan
    assert "Craft copper_dagger" in pb.plan
    assert "## Action" not in pb.plan  # the action is not part of the plan


def test_execute_turn_shows_committed_plan_and_does_not_revise() -> None:
    pb = PlanningPromptBuilder({})
    pb.observe(FIRST_OUTPUT)
    committed = pb.plan

    # Turn 1+: the committed plan is shown back, the model is asked for the next action.
    out = pb.build(state=STATE, retrieval=RET, task="Craft a copper dagger.", history=[])
    assert "## Your committed plan" in out
    assert "gather 8 copper_ore" in out
    assert "Reflect" in out and "Thought" in out  # CoT scaffold on execute turns

    # Stage 2 never re-plans: a later, different plan output is ignored.
    pb.observe('## Plan\n1. A totally different plan.\n## Action\n{"action": "rest", "args": {}}')
    assert pb.plan == committed


def test_observe_without_plan_section_keeps_plan_empty() -> None:
    pb = PlanningPromptBuilder({})
    pb.observe('{"action": "rest", "args": {}}')  # no "## Plan" header
    assert pb.plan == ""


def test_plan_only_output_still_commits() -> None:
    """A model that writes ONLY a plan and stops (no action JSON, no following
    header -- e.g. truncated at max_tokens) must still commit the plan instead
    of being silently re-asked next turn."""
    pb = PlanningPromptBuilder({})
    pb.observe("## Plan\n1. Gather copper_ore.\n2. Smelt copper.")
    assert "Gather copper_ore" in pb.plan and "Smelt copper" in pb.plan
