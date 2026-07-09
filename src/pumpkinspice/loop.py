"""The conventional RAG agent loop (spec section 2, item 4).

Per turn:  get world state -> build retrieval query -> retrieve -> build the
typical RAG prompt -> call the decoder -> parse the action -> act -> capture.

The model decides what to do; the harness does NOT maintain a world model and
does NOT autonomically surface memory. That conventional shape is the point of
the control -- do not "improve" it into autonomic behavior.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from .contracts import (
    Action,
    Capture,
    Decoder,
    PromptBuilder,
    Retrieval,
    Turn,
    World,
)
from .logging import get_logger

log = get_logger("pumpkinspice.loop")


def parse_action(text: str) -> Action:
    """Extract an action from model output.

    Convention: the model emits a JSON object ``{"action": "<verb>", "args":
    {...}}`` (optionally fenced). We take the first such object. If none is
    found, we fall back to a no-op ``rest`` so a malformed turn is recorded
    rather than crashing the run -- the capture preserves the raw output for
    analysis either way.
    """
    decoder = json.JSONDecoder()
    # Try to decode a complete JSON value at each '{'; raw_decode handles nested
    # braces correctly (a regex cannot match balanced brackets).
    i = text.find("{")
    while i != -1:
        try:
            obj, _end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "action" in obj:
            args = obj.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            return Action(kind=str(obj["action"]).strip().lower(), args=args, raw_text=text)
        i = text.find("{", i + 1)
    return Action(kind="rest", args={}, raw_text=text)


def _item_count(state: dict[str, Any], code: str) -> int:
    """Quantity of ``code`` in a HeroBench character state's inventory (list of
    {code, quantity} or a {code: qty} dict)."""
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


def fight_result(outcome: object) -> dict[str, Any] | None:
    """The HeroBench fight sub-response captured in ``outcome["data"]["fight"]``, or None
    if this turn's outcome carries no parseable fight (a non-fight action, an error, or a
    format skew). ``outcome`` is typed ``object`` because it comes from possibly-malformed
    or foreign JSONL captures -- the isinstance guards are load-bearing, not dead code."""
    if not isinstance(outcome, dict):
        return None
    data = outcome.get("data")
    if not isinstance(data, dict):
        return None
    fight = data.get("fight")
    return fight if isinstance(fight, dict) else None


def fight_won_vs(outcome: object, monster: str) -> bool:
    """True if a turn's action was a fight WON against ``monster``. Reads the live HeroBench
    fight response (``result == "win"`` and ``monster == <code>``; the enum is win/lose and
    ``monster`` is always set -- verified against the live server and HeroBench's
    ``FightResult``/endpoint source). The win + fought-monster are authoritative here,
    independent of the ~5%-rate item drop, which is why the v2 capability goal scores on
    this rather than a drop or a level proxy."""
    fight = fight_result(outcome)
    return fight is not None and fight.get("result") == "win" and fight.get("monster") == monster


class AgentLoop:
    def __init__(
        self,
        *,
        decoder: Decoder,
        retrieval: Retrieval,
        world: World,
        prompt: PromptBuilder,
        capture: Capture,
        task: str,
        top_k: int = 5,
        sampler: dict[str, Any] | None = None,
        history_window: int = 0,
        goal_item: str | None = None,
        goal_level: int | None = None,
        goal_skill: str | None = None,
        goal_state_key: str | None = None,
        goal_monster: str | None = None,
    ) -> None:
        self.decoder = decoder
        self.retrieval = retrieval
        self.world = world
        self.prompt = prompt
        self.capture = capture
        self.task = task
        self.top_k = top_k
        self.sampler = sampler or {}
        # Stop-on-goal: when set, the run ends as soon as the goal is reached, so
        # steps == steps-to-completion (not always max_turns). None -> run the full
        # budget (the web's free-form tasks have no machine-checkable goal).
        self.goal_item = goal_item
        self.goal_level = goal_level
        # goal_skill scopes goal_level to a SKILL (e.g. "weaponcrafting" -> the
        # state's weaponcrafting_level) instead of the character's combat level.
        self.goal_skill = goal_skill
        # goal_state_key: success = state[key] is truthy (a World that reports its
        # own solved-ness directly, e.g. HanoiWorld's "solved" -- no item/level
        # counting needed).
        self.goal_state_key = goal_state_key
        # goal_monster: success = the run WON a fight vs this monster code (the v2
        # capability-milestone goal). Unlike the others this is OUTCOME-based (read from
        # the fight response captured in a turn, not world state) -- see fight_won_vs.
        # Normalize a falsy goal_monster ("" from a blanked TOML key) to None, so it does not
        # take the outcome branch below and silently disable a co-set state-based goal.
        self.goal_monster = goal_monster or None
        self._goal_baseline = 0  # count of goal_item at run start (set in play())
        # In-context working memory: the agent's own prior turns, fed back into the
        # prompt. NOT persisted, NOT written back to any store (the agent's DB role
        # is read-only) -- conventional working memory, not autonomic memory.
        # history_window <= 0 means full history (models run at max context, ~200k,
        # so there is no need to truncate); a positive value caps the window.
        self.history_window = history_window
        self._turns: list[Turn] = []

    def run_turn(self, index: int) -> Turn:
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        state = self.world.get_state()
        t["world_get_state"] = (time.perf_counter() - t0) * 1e3

        query = self.prompt.query_for(state=state, task=self.task)

        t0 = time.perf_counter()
        retrieval = self.retrieval.retrieve(query, top_k=self.top_k)
        t["retrieval"] = (time.perf_counter() - t0) * 1e3

        history = self._turns if self.history_window <= 0 else self._turns[-self.history_window :]
        rendered = self.prompt.build(
            state=state, retrieval=retrieval, task=self.task, history=history
        )

        t0 = time.perf_counter()
        try:
            raw = self.decoder.complete(rendered, sampler=self.sampler)
        except Exception as exc:
            # A transient decoder failure (timeout, connection reset, 5xx) must
            # cost one turn, not the whole run: record it as an empty turn (the
            # guard below flags it and the action falls back to `rest`).
            log.error(
                "turn %d: decoder call FAILED (%s: %s) -> recording an empty turn",
                index,
                type(exc).__name__,
                exc,
            )
            raw = ""
        t["decode"] = (time.perf_counter() - t0) * 1e3

        # Empty-content guard: a reasoning model that did not finish thinking
        # returns no content, so the agent would silently fall back to `rest`
        # every turn. Surface it loudly instead of failing silently.
        decoder_empty = not raw.strip()
        if decoder_empty:
            log.warning(
                "turn %d: decoder returned EMPTY output -> falling back to 'rest'. "
                "If this is a reasoning model, it likely did not finish thinking; "
                "increase the decoder's max_tokens.",
                index,
            )

        action = parse_action(raw)
        # Best-effort provenance: the chain-of-thought and model id, if the decoder
        # exposes them (for the reasoning viewer and cross-model analysis).
        reasoning = getattr(self.decoder, "last_reasoning", "")
        model = getattr(self.decoder, "model", "") or ""
        usage = getattr(self.decoder, "last_usage", None) or {}

        # Planning strategies (Stage 2+) are stateful prompt builders that learn a
        # committed plan from the model's output. Optional, duck-typed: the default
        # reactive builder has no `observe`/`plan`, so this is a no-op for it.
        observe = getattr(self.prompt, "observe", None)
        if callable(observe):
            observe(raw)
        plan = str(getattr(self.prompt, "plan", ""))

        t0 = time.perf_counter()
        result = self.world.act(action)
        t["world_act"] = (time.perf_counter() - t0) * 1e3
        # Loud failure on a fight whose response has no parseable fight block: a HeroBench
        # response-format skew would otherwise make goal_monster score every genuine win as a
        # loss SILENTLY (full budget burned, 0% correct, no error). Surface it here.
        if action.kind == "fight" and result.ok and fight_result({"data": result.data}) is None:
            keys = (
                list(result.data) if isinstance(result.data, dict) else type(result.data).__name__
            )
            log.warning(
                "turn %d: fight action returned no parseable fight result (data=%s) -- "
                "verify the HeroBench fight response shape; goal_monster scoring depends on it.",
                index,
                keys,
            )

        # Decode provenance (the experiment's IV record): the request the decoder actually
        # sent this turn (effective sampler incl. seed, max_tokens, model, extra_body such as
        # the enable_thinking no-think flag), minus the prompt. Read from the decoder's
        # snapshot so the record matches the wire; duck-typed so decoders that do not expose
        # it (mock/echo) record empty.
        decode = dict(getattr(self.decoder, "last_request", {}))
        turn = Turn(
            index=index,
            task=self.task,
            world_state=state.raw,
            retrieval={
                "query": retrieval.query,
                "backend": retrieval.backend,
                "latency_ms": retrieval.latency_ms,
                "nodes": [
                    {"id": n.id, "score": n.score, "text": n.text, "metadata": n.metadata}
                    for n in retrieval.nodes
                ],
            },
            prompt=rendered,
            raw_output=raw,
            action={"kind": action.kind, "args": action.args},
            outcome={
                "ok": result.ok,
                "status_code": result.status_code,
                "error": result.error,
                "data": result.data,
            },
            timings_ms=t,
            decoder_empty=decoder_empty,
            reasoning=reasoning,
            model=model,
            plan=plan,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            decode=decode,
        )
        self.capture.record(turn)
        self._turns.append(turn)
        return turn

    def _goal_reached(self) -> bool:
        """Has the run reached its goal? Checked after each turn (play breaks on the first
        True). Exactly one goal_* should be set; if several are, PRECEDENCE is
        goal_monster > goal_state_key > goal_item > goal_level (first set wins).

        - goal_monster (v2, OUTCOME-based): the just-played turn WON a fight vs the target
          monster (read from the captured fight response, not world state). Only the last
          turn can newly satisfy it -- earlier turns were already checked and did not stop.
        - goal_item: the agent CRAFTED the item this run (count > the start baseline), not
          merely that it is present -- a reset character can carry residual inventory, which
          would otherwise read as an instant false completion.
        - goal_level / goal_skill / goal_state_key: absolute state checks (you cannot un-level;
          a fresh get_state is the reliable post-action source)."""
        if (
            self.goal_item is None
            and self.goal_level is None
            and self.goal_state_key is None
            and self.goal_monster is None
        ):
            return False
        if self.goal_monster is not None:
            # Only the just-appended turn can newly satisfy this (O(1), not O(n) per call).
            return bool(self._turns) and fight_won_vs(self._turns[-1].outcome, self.goal_monster)
        try:
            state = self.world.get_state().raw
        except Exception:  # never let a goal check abort an otherwise-fine run
            return False
        if self.goal_item is not None:
            return _item_count(state, self.goal_item) > self._goal_baseline
        if self.goal_state_key is not None:
            return bool(state.get(self.goal_state_key))
        key = f"{self.goal_skill}_level" if self.goal_skill else "level"
        lvl = state.get(key)
        return isinstance(lvl, int) and self.goal_level is not None and lvl >= self.goal_level

    def play(self, max_turns: int, should_stop: Callable[[], bool] | None = None) -> list[Turn]:
        # Baseline the goal item BEFORE playing, so completion means "crafted this run"
        # -- robust to a contaminated start.
        self._goal_baseline = 0
        if self.goal_item is not None:
            try:
                self._goal_baseline = _item_count(self.world.get_state().raw, self.goal_item)
            except Exception:
                self._goal_baseline = 0
        try:
            for i in range(max_turns):
                if should_stop is not None and should_stop():
                    log.info("run stopped by request at turn %d (%d turns played)", i, i)
                    break
                self.run_turn(i)
                if self._goal_reached():
                    log.info("goal reached at turn %d -- stopping early (%d turns)", i, i + 1)
                    break
        finally:
            self.capture.close()
        return self._turns
