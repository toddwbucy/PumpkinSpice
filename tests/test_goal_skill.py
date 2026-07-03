"""Skill-scoped level goals ("reach weaponcrafting 5"): the calibrated task needs
success/stop checks on a SKILL level, not just an item craft or character level."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pumpkinspice import kernel
from pumpkinspice.analyze import analyze_turns
from pumpkinspice.contracts import Action, ActionResult, WorldState
from pumpkinspice.loop import AgentLoop
from pumpkinspice.web.app import _apply_goal


def _turn(wc_level: int) -> dict[str, Any]:
    return {
        "index": 0,
        "world_state": {"x": 0, "y": 0, "level": 1, "weaponcrafting_level": wc_level},
        "retrieval": {},
        "action": {"kind": "craft", "args": {}},
        "outcome": {"ok": True},
        "timings_ms": {"decode": 1.0},
    }


def test_analyze_success_on_skill_level() -> None:
    turns = [_turn(1), _turn(5)]
    m = analyze_turns("r", turns, goal_level=5, goal_skill="weaponcrafting")
    assert m.success is True
    # character level stays 1: a bare goal_level=5 must NOT read as success
    assert analyze_turns("r", turns, goal_level=5).success is False
    assert (
        analyze_turns("r", [_turn(1), _turn(4)], goal_level=5, goal_skill="weaponcrafting").success
        is False
    )


class _LevelingWorld:
    """weaponcrafting_level rises by 1 per action; reaches 3 after 2 acts."""

    name = "leveling"

    def __init__(self) -> None:
        self.acts = 0

    def get_state(self) -> WorldState:
        return WorldState(raw={"x": 0, "y": 0, "level": 1, "weaponcrafting_level": 1 + self.acts})

    def act(self, action: Action) -> ActionResult:
        self.acts += 1
        return ActionResult(ok=True, status_code=200)


class _ActDecoder:
    name = "act"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "craft", "args": {"code": "copper_dagger", "quantity": 1}}'


def test_loop_stops_on_skill_level(tmp_path: Path) -> None:
    loop = AgentLoop(
        decoder=_ActDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=_LevelingWorld(),  # type: ignore[arg-type]
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="reach weaponcrafting 3",
        goal_level=3,
        goal_skill="weaponcrafting",
    )
    turns = loop.play(10)
    assert len(turns) == 2  # wc hits 3 after the 2nd act -> early stop


def test_apply_goal_spec_parsing() -> None:
    run: dict[str, Any] = {}
    _apply_goal(run, "weaponcrafting_level>=5")
    assert run == {"goal_level": 5, "goal_skill": "weaponcrafting"}
    run = {}
    _apply_goal(run, "level>=4")
    assert run == {"goal_level": 4}
    run = {}
    _apply_goal(run, "sticky_sword")
    assert run == {"goal_item": "sticky_sword"}
    run = {}
    _apply_goal(run, None)
    _apply_goal(run, "  ")
    assert run == {}


def test_analyze_success_on_state_key() -> None:
    """goal_state_key: success = state[key] is truthy (HanoiWorld's "solved",
    or any World that self-reports goal state directly)."""

    def hturn(solved: bool) -> dict[str, Any]:
        return {
            "index": 0,
            "world_state": {"pegs": {"A": [], "B": [], "C": []}, "solved": False},
            "retrieval": {},
            "action": {"kind": "move", "args": {}},
            "outcome": {"ok": True, "data": {"pegs": {"C": [1]}, "solved": solved}},
            "timings_ms": {"decode": 1.0},
        }

    assert analyze_turns("r", [hturn(False), hturn(True)], goal_state_key="solved").success is True
    assert (
        analyze_turns("r", [hturn(False), hturn(False)], goal_state_key="solved").success is False
    )
    # bare goal_level (no goal_state_key) must not be tripped by an unrelated "solved" key
    assert analyze_turns("r", [hturn(True)], goal_level=5).success is False


class _HanoiLikeWorld:
    """Reports solved=True immediately after the first action (stand-in for
    HanoiWorld, without depending on it -- keeps this a loop.py unit test)."""

    name = "hanoi-like"

    def __init__(self) -> None:
        self.acts = 0

    def get_state(self) -> WorldState:
        return WorldState(raw={"solved": self.acts >= 1})

    def act(self, action: Action) -> ActionResult:
        self.acts += 1
        return ActionResult(ok=True, status_code=200)


def test_loop_stops_on_state_key(tmp_path: Path) -> None:
    loop = AgentLoop(
        decoder=_ActDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=_HanoiLikeWorld(),  # type: ignore[arg-type]
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="solve it",
        goal_state_key="solved",
    )
    turns = loop.play(10)
    assert len(turns) == 1  # solved after the 1st act -> immediate stop
