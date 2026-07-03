"""Stage-3 strategy: plan + replan-on-surprise.

Like the Stage-2 plan (commit a plan, execute it on the shared Reflexion+ReAct CoT
[[prompt_default]]), but the plan ADAPTS. When the world surprises the agent -- a
failed action, a discovery that contradicts the plan -- it may rewrite the plan:
`observe` updates the committed plan whenever the model emits a new "## Plan" (not
only on turn 0), and the prompt invites a revision after a failure. This is the
combination the Stage-1/2 result argued for: a plan gives goal-direction, reactive
replanning gives adaptivity, and neither alone completed. Still conventional RAG
(no autonomic memory, no written-back world model).

The plan is captured per turn (`Turn.plan`); a turn where it changes is a *replan*,
which `analyze` counts (the Stage-3 metric).
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import RetrievalResult, Turn, WorldState
from .prompt_default import ACTION_GRAMMAR, query_for, render_history, render_notes

SYSTEM_REPLAN = f"""\
You are a capable agent playing HeroBench, an RPG-style planning environment.

{ACTION_GRAMMAR}

On the FIRST turn, write a PLAN: a numbered list of concrete steps to accomplish the
Goal, derived from the recipes, sources, and locations in the Reference notes.

Your plan is a LIVING plan. On EVERY turn, reason in three labeled steps, then act:

Reflect: Look at your most recent action and where you are in your plan. Did it
  succeed? If it FAILED, the world has contradicted your plan -- diagnose why (wrong
  tile, missing prerequisite item/level, non-adjacent move) and DECIDE: keep the plan,
  or revise it. (First turn: "first turn".)
Thought: Pick the single best LEGAL action toward your current step. POSITION RULE: to
  gather or craft you must be standing EXACTLY ON the resource/workshop tile -- if your
  (x,y) is not the target, MOVE there first (one adjacent step per turn). Do not repeat
  an action that just failed for the same reason.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "<verb>", "args": {{ ... }}}}

To REVISE the plan, write an updated "## Plan" (numbered steps) BEFORE your Reflect.
Revise only when the world has actually contradicted the plan -- do not churn it.
"""

# Pull the "## Plan ..." block out of an output (turn 0 or a later revision).
_PLAN_RE = re.compile(r"##\s*Plan\b[ \t]*\n?(.*?)(?:\n##\s|\n*\{)", re.DOTALL | re.IGNORECASE)


class ReplanPromptBuilder:
    name = "replan"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._system = config.get("system", SYSTEM_REPLAN)
        self.plan = ""

    def query_for(self, *, state: WorldState, task: str) -> str:
        return query_for(state, task)

    def observe(self, raw: str) -> None:
        """Update the living plan whenever the model writes a "## Plan" -- on turn 0
        OR on any later replan. (Stage 2 commits the first plan and never revises;
        Stage 3 differs precisely here.)"""
        m = _PLAN_RE.search(raw)
        if m and m.group(1).strip():
            self.plan = m.group(1).strip()

    def build(
        self,
        *,
        state: WorldState,
        retrieval: RetrievalResult,
        task: str,
        history: list[Turn],
    ) -> str:
        notes = render_notes(retrieval)
        state_json = json.dumps(state.raw, indent=2)
        if not self.plan:
            # PLAN turn: no plan yet -> ask for one, then the first turn.
            return (
                f"{self._system}\n"
                f"## Goal\n{task}\n\n"
                f"## World state\n{state_json}\n\n"
                f"## Reference notes\n{notes}\n\n"
                f'## Now: write "## Plan" (numbered steps), then "## Thought", then "## Action"\n'
            )
        # Surprise nudge: if the last action failed, prompt an explicit keep-or-revise.
        nudge = ""
        if history and not history[-1].outcome.get("ok", True):
            nudge = (
                "## NOTE: your last action FAILED -- the world differs from your plan. In "
                "Reflect, decide whether to REVISE (write an updated ## Plan first) or continue.\n\n"
            )
        return (
            f"{self._system}\n"
            f"## Goal\n{task}\n\n"
            f"## Your current plan (revise only if the world contradicts it)\n{self.plan}\n\n"
            f"## Recent actions (most recent last)\n{render_history(history)}\n\n"
            f"## World state\n{state_json}\n\n"
            f"## Reference notes\n{notes}\n\n"
            f"{nudge}"
            f"## Your turn -- (revise the plan only if needed), then Reflect, Thought, Action\n"
        )
