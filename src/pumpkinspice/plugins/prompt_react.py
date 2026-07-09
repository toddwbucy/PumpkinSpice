"""Externalized-reasoning strategy (floor-test v2): bounded ReAct steps, thinking OFF.

The v2 "reasoning location" independent variable. Where [[prompt_default]]/plan/replan
expect the model to reason INTERNALLY (a long <think> block whose length confounds the
trajectory geometry -- the v1 length finding), this strategy externalizes reasoning into
the harness: pair it with a no-think decoder (`enable_thinking=False`) and a small
`max_tokens`, so each turn is a SHORT, bounded observe->plan->act step whose reasoning is
visible and roughly uniform in length. History is the working memory (conventional RAG; no
persisted or written-back state).

ICL-free by construction (ICL is deferred to WeaverTools): the model plans COLD from the
observed state, with no worked example in context. The measured unit is the first turn's
trajectory. See docs/floor-test-v2-design.md.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import RetrievalResult, Turn, WorldState
from .prompt_default import ACTION_GRAMMAR, query_for, render_history, render_notes

SYSTEM_REACT = f"""\
You are a capable agent playing HeroBench, an RPG-style planning environment.

{ACTION_GRAMMAR}

Reason EXTERNALLY and BRIEFLY -- a few short lines, not an extended analysis. Each turn:

Plan: (first turn only) a short numbered list of concrete steps toward the Goal, drawn from
  the recipes, sources, and locations in the Reference notes.
Thought: one or two sentences picking the single best LEGAL action toward your plan.
  POSITION RULE: to gather or craft you must stand EXACTLY ON the resource/workshop tile --
  if your (x,y) is not the target, MOVE there first (one adjacent step per turn). Do not
  repeat an action that just failed for the same reason.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "<verb>", "args": {{ ... }}}}
"""


class ReactPromptBuilder:
    name = "react"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._system = config.get("system", SYSTEM_REACT)

    def query_for(self, *, state: WorldState, task: str) -> str:
        return query_for(state, task)

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
        if not history:
            # First turn (the MEASURED one): elicit a brief plan + the first action, cold.
            return (
                f"{self._system}\n"
                f"## Goal\n{task}\n\n"
                f"## World state\n{state_json}\n\n"
                f"## Reference notes\n{notes}\n\n"
                '## Now: brief "Plan", then "Thought", then "Action"\n'
            )
        nudge = ""
        if not history[-1].outcome.get("ok", True):
            nudge = (
                "## NOTE: your last action FAILED -- diagnose briefly and pick a different "
                "legal action.\n\n"
            )
        return (
            f"{self._system}\n"
            f"## Goal\n{task}\n\n"
            f"## Recent actions (most recent last)\n{render_history(history)}\n\n"
            f"## World state\n{state_json}\n\n"
            f"## Reference notes\n{notes}\n\n"
            f"{nudge}"
            '## Your turn -- brief "Thought", then "Action"\n'
        )
