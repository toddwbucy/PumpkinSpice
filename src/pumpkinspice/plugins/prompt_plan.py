"""Stage-2 planning strategy: a committed plan layered on the shared CoT.

Same Reflexion + ReAct method as the reactive baseline ([[prompt_default]]), plus a
multi-turn backbone: on turn 0 the model writes a numbered plan; it is parsed out
(`observe`) and COMMITTED -- held fixed for the run (no re-planning; that is Stage
3) -- and shown back each turn while the model Reflects/Thinks/Acts toward the
current step. The plan is the only thing added over the reactive stage, so the two
are directly comparable. Captured as `Turn.plan`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import RetrievalResult, Turn, WorldState
from .prompt_default import ACTION_GRAMMAR, query_for, render_history, render_notes

SYSTEM_PLAN = f"""\
You are a capable agent playing HeroBench, an RPG-style planning environment.

{ACTION_GRAMMAR}

On the FIRST turn, write a PLAN: a numbered list of concrete steps to accomplish the
Goal, derived from the recipes, sources, and locations in the Reference notes.

On EVERY turn, reason in THREE labeled steps, then act:

Reflect: Look at your most recent action and where you are in your committed plan.
  Did the last action succeed? If it FAILED, diagnose WHY in concrete game terms
  (wrong tile, missing prerequisite item/level, non-adjacent move, no workshop here)
  and what you will change. State which plan step you are on. (First turn: "first turn".)
Thought: Pick the single best LEGAL action toward the CURRENT plan step. POSITION RULE:
  to gather or craft you must be standing EXACTLY ON the resource/workshop tile -- an
  adjacent tile is NOT enough. If your (x,y) is not already the target tile, MOVE toward
  it first (one adjacent step per turn) and only gather/craft once on it. Do not repeat
  an action that just failed.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "<verb>", "args": {{ ... }}}}

Do NOT rewrite the plan once committed -- follow it.
"""

# Pull the "## Plan ..." block out of the first output: everything between the Plan
# header and the next "##" header (e.g. "## Thought") or the start of the action JSON.
# The terminator alternation includes end-of-string (\Z): a model that writes
# only a plan and stops (e.g. truncated at max_tokens) must still commit it.
_PLAN_RE = re.compile(r"##\s*Plan\b[ \t]*\n?(.*?)(?:\n##\s|\n*\{|\Z)", re.DOTALL | re.IGNORECASE)


class PlanningPromptBuilder:
    name = "plan"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._system = config.get("system", SYSTEM_PLAN)
        self.plan = ""  # committed once the model produces one on turn 0

    def query_for(self, *, state: WorldState, task: str) -> str:
        return query_for(state, task)

    def observe(self, raw: str) -> None:
        """Learn and COMMIT the plan from the model's first output (Stage 2: the
        first plan only; never revised). Called by the loop after each decode."""
        if self.plan:
            return
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
            # PLAN turn: no committed plan yet -> ask for one, then the first turn.
            return (
                f"{self._system}\n"
                f"## Goal\n{task}\n\n"
                f"## World state\n{state_json}\n\n"
                f"## Reference notes\n{notes}\n\n"
                f'## Now: write "## Plan" (numbered steps), then "## Thought", then "## Action"\n'
            )
        # EXECUTE turn: committed plan is the stable backbone; volatile state/notes last.
        return (
            f"{self._system}\n"
            f"## Goal\n{task}\n\n"
            f"## Your committed plan (follow it; do NOT rewrite it)\n{self.plan}\n\n"
            f"## Recent actions (most recent last)\n{render_history(history)}\n\n"
            f"## World state\n{state_json}\n\n"
            f"## Reference notes\n{notes}\n\n"
            f"## Your turn -- Reflect (on the last outcome and your plan step), Thought, Action\n"
        )
