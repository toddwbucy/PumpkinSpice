"""Offline world: a minimal in-memory HeroBench stand-in.

Accepts the documented action verbs and mutates a tiny character state so the
loop produces non-trivial captures without a live HeroBench server. Not a
simulator -- just enough to exercise the harness.
"""

from __future__ import annotations

from typing import Any

from ..contracts import Action, ActionResult, WorldState


class MockWorld:
    name = "mock"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._state: dict[str, Any] = {
            "name": config.get("character", "hero"),
            "x": 0,
            "y": 0,
            "level": 1,
            "xp": 0,
            "hp": 100,
            "max_hp": 100,
            "inventory": {},
        }

    def get_state(self) -> WorldState:
        return WorldState(raw=dict(self._state), source=self.name)

    def act(self, action: Action) -> ActionResult:
        s = self._state
        k = action.kind
        a = action.args
        if k == "move":
            s["x"], s["y"] = int(a.get("x", s["x"])), int(a.get("y", s["y"]))
        elif k == "fight":
            s["xp"] += 10
            s["hp"] = max(0, s["hp"] - 5)
            if s["xp"] >= s["level"] * 50:
                s["level"] += 1
        elif k in ("gather", "gathering"):
            code = a.get("code", "ash_wood")
            s["inventory"][code] = s["inventory"].get(code, 0) + int(a.get("quantity", 1))
        elif k in ("craft", "crafting"):
            code = a.get("code", "item")
            s["inventory"][code] = s["inventory"].get(code, 0) + int(a.get("quantity", 1))
        elif k == "rest":
            s["hp"] = s["max_hp"]
        elif k == "equip":
            s.setdefault("equipment", {})[a.get("slot", "weapon")] = a.get("code", "")
        else:
            return ActionResult(ok=False, status_code=400, error=f"unknown action {k!r}")
        return ActionResult(ok=True, status_code=200, data=dict(s))
