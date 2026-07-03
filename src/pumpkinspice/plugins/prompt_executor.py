"""Stage-4 strategy: plan-and-execute (the harness holds the plan).

The Stage-1..3 result: models can ARTICULATE the right plan (gemma wrote "craft
copper_daggers to level weaponcrafting to 5") but never EXECUTE it -- free-text
plans re-rendered each turn do not survive re-prompting, and the locally-cheap
action always wins over step 3 of a half-remembered plan. This builder closes
that gap with standard plan-and-execute scaffolding: the model writes its plan
ONCE as strict JSON with a machine-checkable ``done_when`` per step; the
*harness* stores the plan, advances steps mechanically against world state, and
each turn hands back exactly ONE current step to act toward.

Still conventional RAG: the plan is model-authored, lives only for this run
(same lifetime as the in-context history window), and nothing is persisted or
written back. This is a competent practitioner's agent scaffold (the "no
sandbagging" clause), not autonomic behavior.

The committed plan is captured per turn (``Turn.plan``, descriptions only, so a
step ADVANCE is not a replan); a turn where it changes counts as a replan in
`analyze`, comparable with Stage 3.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts import RetrievalResult, Turn, WorldState
from .prompt_default import ACTION_GRAMMAR, query_for, render_history, render_notes

SYSTEM_EXECUTOR = f"""\
You are a capable agent playing HeroBench, an RPG-style planning environment.

{ACTION_GRAMMAR}

You work with a PLAN EXECUTOR: you write your plan ONCE as a JSON object; the
harness stores it, tracks the current step, and advances a step automatically
the moment its done_when condition holds in the world state. Each turn you are
shown ONLY the current step -- act toward THAT step; do not re-derive the plan.

Plan format (a single JSON object on its own line):
{{"plan": [{{"step": 1, "description": "<one concrete objective>", "done_when": {{...}}}}, ...]}}

Every step MUST carry a machine-checkable done_when using ONLY these forms
(combine them; every listed condition must hold for the step to complete):
  {{"inventory": {{"<item_code>": <min_quantity>}}}}   e.g. {{"inventory": {{"copper_ore": 40}}}}
  {{"skill": {{"<skill_name>": <min_level>}}}}         e.g. {{"skill": {{"weaponcrafting": 5}}}}
  {{"position": [<x>, <y>]}}                          e.g. {{"position": [2, 0]}}
Make steps OUTCOME-sized ("gather 40 copper_ore at (2,0)"), not single moves.

Each turn, reason briefly, then act:
Reflect: did your last action succeed? If it FAILED, diagnose why in concrete
  game terms (wrong tile, missing prerequisite item/level, non-adjacent move).
Action: output exactly ONE action as a JSON object on its own line, then STOP:
  {{"action": "<verb>", "args": {{ ... }}}}
POSITION RULE: to gather or craft you must stand EXACTLY ON the resource or
workshop tile -- if your (x,y) is not the target, move first (one adjacent step
per turn). Do not repeat an action that just failed for the same reason.
If the CURRENT step is already complete but was not auto-advanced (its
done_when was mis-specified), add "step_done": true to your action JSON and act
toward the NEXT step instead.
Rewrite the plan (a new {{"plan": [...]}} line before your action) ONLY when the
world has genuinely contradicted it or you are explicitly told to revise.
"""


def _count(state: dict[str, Any], code: str) -> int:
    """Quantity of ``code`` in a HeroBench inventory (list of {code, quantity})."""
    inv = state.get("inventory")
    if isinstance(inv, list):
        return sum(
            int(i.get("quantity") or 0)
            for i in inv
            if isinstance(i, dict) and i.get("code") == code
        )
    if isinstance(inv, dict):
        return int(inv.get(code, 0) or 0)
    return 0


def _condition_met(cond: dict[str, Any], state: dict[str, Any]) -> bool:
    """True when EVERY recognized clause of ``cond`` holds in ``state``. A step
    with no recognized clause never auto-advances (the model can still advance
    it explicitly with "step_done", or revise the plan)."""
    recognized = False
    inv = cond.get("inventory")
    if isinstance(inv, dict) and inv:
        recognized = True
        for code, qty in inv.items():
            try:
                need = int(qty)
            except (TypeError, ValueError):
                return False
            if _count(state, str(code)) < need:
                return False
    skill = cond.get("skill")
    if isinstance(skill, dict) and skill:
        recognized = True
        for name, lvl in skill.items():
            key = "level" if str(name) in ("level", "character") else f"{name}_level"
            cur = state.get(key)
            try:
                need = int(lvl)
            except (TypeError, ValueError):
                return False
            if not isinstance(cur, int) or cur < need:
                return False
    pos = cond.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        recognized = True
        if state.get("x") != pos[0] or state.get("y") != pos[1]:
            return False
    # Domain-agnostic escape hatch: exact-match any top-level state key. Lets a
    # non-HeroBench World (e.g. HanoiWorld's "solved"/"pegs") express a
    # machine-checkable done_when without teaching this module its state shape.
    st = cond.get("state")
    if isinstance(st, dict) and st:
        recognized = True
        for key, want in st.items():
            if state.get(key) != want:
                return False
    return recognized


def _parse_plan(text: str) -> list[dict[str, Any]] | None:
    """First ``{"plan": [...]}`` JSON object in ``text`` (same balanced-brace
    scan as loop.parse_action), normalized to numbered steps. None if absent or
    it contains no usable step."""
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
            steps: list[dict[str, Any]] = []
            for s in obj["plan"]:
                if isinstance(s, dict) and str(s.get("description") or "").strip():
                    done = s.get("done_when")
                    steps.append(
                        {
                            "step": len(steps) + 1,
                            "description": str(s["description"]).strip(),
                            "done_when": done if isinstance(done, dict) else {},
                        }
                    )
            return steps or None
        i = text.find("{", i + 1)
    return None


def _turn_failed(turn: Turn) -> bool:
    """A turn counts as failed for the stuck-streak when the action errored OR a
    fight was LOST -- HeroBench returns HTTP 200 for a lost fight (result:
    "lose", no drops), which must not read as progress."""
    if not turn.outcome.get("ok", True):
        return True
    fight = (turn.outcome.get("data") or {}).get("fight")
    return isinstance(fight, dict) and fight.get("result") == "lose"


def _step_done_flag(text: str) -> bool:
    """True when the action JSON in ``text`` carries ``"step_done": true``."""
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj.get("step_done") is True
        i = text.find("{", i + 1)
    return False


class ExecutorPromptBuilder:
    name = "executor"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._system = config.get("system", SYSTEM_EXECUTOR)
        # After this many CONSECUTIVE failed actions, the prompt demands a revision.
        self._replan_after = int(config.get("replan_after", 3))
        self._steps: list[dict[str, Any]] = []
        self._idx = 0
        # Captured per turn (Turn.plan). Descriptions only -- advancing a step must
        # NOT read as a replan; only an actual rewrite changes this string.
        self.plan = ""

    # -- plan state ------------------------------------------------------------

    def _current(self) -> dict[str, Any] | None:
        return self._steps[self._idx] if self._idx < len(self._steps) else None

    def _advance(self, state: dict[str, Any]) -> None:
        while self._idx < len(self._steps) and _condition_met(
            self._steps[self._idx].get("done_when", {}), state
        ):
            self._idx += 1

    def observe(self, raw: str) -> None:
        """Learn a (re)written plan from the model's output; otherwise honor an
        explicit "step_done" advance. A fresh plan restarts at step 1 (mechanical
        advance immediately skips any steps the world already satisfies)."""
        steps = _parse_plan(raw)
        if steps:
            self._steps = steps
            self._idx = 0
            self.plan = "\n".join(f"{s['step']}. {s['description']}" for s in steps)
            return
        if self._current() is not None and _step_done_flag(raw):
            self._idx += 1

    # -- PromptBuilder contract --------------------------------------------------

    def query_for(self, *, state: WorldState, task: str) -> str:
        self._advance(state.raw)
        step = self._current()
        if step is not None:
            # Retrieval targets the CURRENT step (its recipes/locations), not the
            # whole task -- the step is what the model must act on now.
            return query_for(state, f"{task} -- current step: {step['description']}")
        return query_for(state, task)

    def build(
        self,
        *,
        state: WorldState,
        retrieval: RetrievalResult,
        task: str,
        history: list[Turn],
    ) -> str:
        self._advance(state.raw)
        notes = render_notes(retrieval)
        state_json = json.dumps(state.raw, indent=2)

        step = self._current()
        if step is None:
            # PLAN turn: no plan yet, or every step is done but the goal is not.
            exhausted = (
                "## NOTE: every step of your previous plan is complete, but the Goal "
                "is NOT yet achieved. Your plan was missing something -- write a NEW plan.\n\n"
                if self._steps
                else ""
            )
            return (
                f"{self._system}\n"
                f"## Goal\n{task}\n\n"
                f"## World state\n{state_json}\n\n"
                f"## Reference notes\n{notes}\n\n"
                f"{exhausted}"
                f'## Now: write your {{"plan": [...]}} JSON (each step with a checkable '
                f"done_when), then your FIRST action JSON\n"
            )

        # EXECUTE turn: hand back exactly one step.
        lines = []
        for j, s in enumerate(self._steps):
            mark = "[done]" if j < self._idx else ("[NOW] " if j == self._idx else "[    ]")
            lines.append(f"{s['step']}. {mark} {s['description']}")
        # Consecutive trailing failures (incl. LOST fights) -> the step or plan is
        # wrong; demand a revision.
        stuck = 0
        for t in reversed(history):
            if not _turn_failed(t):
                break
            stuck += 1
        nudge = ""
        if stuck >= self._replan_after:
            nudge = (
                f"## NOTE: your last {stuck} actions ALL FAILED. The plan or the current "
                f'step is wrong. REVISE NOW: write an updated {{"plan": [...]}} JSON '
                f"(checkable done_when per step) before your action.\n\n"
            )
        return (
            f"{self._system}\n"
            f"## Goal\n{task}\n\n"
            f"## Your plan (held by the harness; [NOW] = current step)\n"
            + "\n".join(lines)
            + "\n\n"
            f"## CURRENT STEP {step['step']} of {len(self._steps)}: {step['description']}\n"
            f"done_when: {json.dumps(step.get('done_when') or {})}\n\n"
            f"## Recent actions (most recent last)\n{render_history(history)}\n\n"
            f"## World state\n{state_json}\n\n"
            f"## Reference notes\n{notes}\n\n"
            f"{nudge}"
            f"## Your turn -- Reflect briefly, then ONE action JSON toward the CURRENT step\n"
        )
