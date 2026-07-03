"""The default reasoning method: a Reflexion + ReAct chain-of-thought.

This is the agent's BASE reasoning scaffold, shared by every stage (the reactive
baseline here; Stage 2's `plan` builder layers a committed plan on top). It is a
*method*, not a task script: the structure (Reflect -> Thought -> Action) is fixed
and game-agnostic, while the content is filled at runtime from three sources --
the task (goal), the world state (what is true now), and retrieval (the game's
knowledge: recipes, sources, locations, prerequisites). Nothing is specific to any
one task; "craft a dagger", "reach level 5", "gather 20 ash" all flow through it.
The only HeroBench-specific surface is the action grammar (the game's API); swap
that plus the retrieval corpus and the same CoT drives a different environment.

Fairness (spec section 6): this is the strongest *fair* conventional pattern -- a
competent practitioner's reasoning scaffold, not a strawman. It is part of the
experiment (one of two things that differ from Agent 1), so tune it deliberately.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import RetrievalResult, Turn, WorldState

# The game's action interface -- the only HeroBench-specific part of the scaffold.
ACTION_GRAMMAR = """\
Actions and their args:
  move:   {"x": <int>, "y": <int>}   -- ADJACENT tiles only (|dx| <= 1 AND |dy| <= 1).
          To reach a far tile, step toward it one tile per turn.
  fight:  {}                          -- fight the monster on your current tile.
  gather: {"quantity": <int>}         -- gather the resource on your current tile.
  craft:  {"code": "<item>", "quantity": <int>}   -- at the matching workshop, with the
          ingredient items already in your inventory.
  equip:  {"slot": "<slot>", "code": "<item>"}
  rest:   {}                          -- recover HP."""

# The Reflexion + ReAct method: reflect on the last outcome, reason to a subgoal,
# then act. The Reflect step is what attacks reactive thrashing -- it forces the
# model to diagnose WHY the last action failed before choosing the next one.
SYSTEM_COT = f"""\
You are a capable agent playing HeroBench, an RPG-style planning environment.

{ACTION_GRAMMAR}

Reason every turn in THREE labeled steps, then act:

Reflect: Look at your most recent action in "Recent actions". Did it succeed? If it
  FAILED, diagnose WHY in concrete game terms -- e.g. you were not on the required
  tile, you lacked a prerequisite item or level, you tried to move more than one
  tile, or the workshop/resource is not where you stand -- and say what you will
  change. If there is no prior action, write "first turn, no prior action".
Thought: Using your Goal, the World state, and the Reference notes, name the
  immediate SUBGOAL (the nearest unmet prerequisite) and the single best LEGAL action
  toward it. POSITION RULE: to gather or craft you must be standing EXACTLY ON the
  resource/workshop tile -- an adjacent tile is NOT enough. If your (x,y) does not
  already equal the target tile's coordinates, MOVE toward it first (one adjacent step
  per turn, changing x and y by at most 1 each) and only gather/craft once your
  position equals the target. Do not repeat an action that just failed for the same reason.
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "<verb>", "args": {{ ... }}}}
"""


def render_notes(retrieval: RetrievalResult) -> str:
    if not retrieval.nodes:
        return "(no reference notes retrieved)"
    return "\n".join(f"- [{n.id} | score={n.score:.3f}] {n.text}" for n in retrieval.nodes)


def _outcome_detail(outcome: dict[str, Any]) -> str:
    """The salient part of the world's response to an action -- fight results
    (win/LOSE, xp, drops) and craft/gather yields (items made, skill xp). This is
    the agent's own action feedback; hiding it would sandbag the control (an
    HTTP-200 LOST fight otherwise renders as a bare "ok"), and it is the only
    channel through which game mechanics like "crafting grants skill xp" are
    discoverable (the encyclopedia does not document them)."""
    data = outcome.get("data") or {}
    fight = data.get("fight")
    if isinstance(fight, dict):
        drops = (
            ", ".join(
                f"{d.get('quantity')}x {d.get('code')}"
                for d in fight.get("drops") or []
                if isinstance(d, dict)
            )
            or "nothing"
        )
        result = str(fight.get("result") or "?").upper()
        return f"fight {result} vs {fight.get('monster', '?')} (xp {fight.get('xp', 0)}, drops: {drops})"
    details = data.get("details")
    if isinstance(details, dict) and details.get("skill"):
        items = ", ".join(
            f"{i.get('quantity')}x {i.get('code')}"
            for i in details.get("items") or []
            if isinstance(i, dict)
        )
        return f"got {items or '?'} (+{details.get('xp', 0)} {details.get('skill')} xp)"
    return ""


def render_history(history: list[Turn]) -> str:
    """A compact log of the agent's own prior turns, so it can Reflect on failures
    and remember where it has already been. Includes each action's salient world
    feedback (fight result, xp gained, drops) -- see _outcome_detail."""
    if not history:
        return "(no actions yet)"
    lines = []
    for t in history:
        act = f"{t.action.get('kind')} {t.action.get('args') or ''}".strip()
        oc = t.outcome
        pos = (t.world_state.get("x"), t.world_state.get("y"))
        outcome = "ok" if oc.get("ok") else f"FAILED ({oc.get('status_code')} {oc.get('error')})"
        detail = _outcome_detail(oc) if oc.get("ok") else ""
        lines.append(
            f"- turn {t.index} (was at {pos}): {act} -> {outcome}"
            + (f" -- {detail}" if detail else "")
        )
    return "\n".join(lines)


def query_for(state: WorldState, task: str) -> str:
    # A competent dev's retrieval query: the task plus salient state cues.
    loc = f"at ({state.raw.get('x')}, {state.raw.get('y')})" if state.raw else ""
    lvl = f"level {state.raw.get('level')}" if state.raw else ""
    return " ".join(p for p in [task, lvl, loc] if p).strip()


class DefaultPromptBuilder:
    """Reactive baseline: the Reflexion + ReAct CoT applied fresh each turn (no
    committed plan -- the model re-reasons every turn)."""

    name = "default"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._system = config.get("system", SYSTEM_COT)

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
        # Order is KV-cache-friendly: stable + append-only content (system, goal,
        # history) first; volatile this-turn content (state, notes) last.
        return (
            f"{self._system}\n"
            f"## Goal\n{task}\n\n"
            f"## Recent actions (most recent last)\n{render_history(history)}\n\n"
            f"## World state\n{json.dumps(state.raw, indent=2)}\n\n"
            f"## Reference notes\n{render_notes(retrieval)}\n\n"
            f"## Your turn -- Reflect, then Thought, then Action\n"
        )
