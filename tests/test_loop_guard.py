"""The empty-content guard: an empty decode warns and is flagged in the capture."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from pumpkinspice import kernel
from pumpkinspice.contracts import Action, ActionResult, WorldState
from pumpkinspice.loop import AgentLoop, fight_won_vs


class _EmptyDecoder:
    name = "empty"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return "   "  # blank, as a reasoning model that ran out of thinking budget


class _ActDecoder:
    name = "act"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "gather", "args": {}}'


class _GoalWorld:
    """get_state returns the goal item only after 2 actions, so the loop should stop
    at turn index 1 (2 turns) rather than running the full budget."""

    name = "goalworld"

    def __init__(self) -> None:
        self.acts = 0

    def get_state(self) -> WorldState:
        inv = [{"code": "copper_dagger", "quantity": 1}] if self.acts >= 2 else []
        return WorldState(raw={"x": 0, "y": 0, "level": 1, "inventory": inv})

    def act(self, action: Action) -> ActionResult:
        self.acts += 1
        return ActionResult(ok=True, status_code=200)


class _ContaminatedWorld:
    """A reset character that ALREADY carries the goal item -- the loop must not read
    this as an instant completion (it was not crafted this run)."""

    name = "contam"

    def get_state(self) -> WorldState:
        return WorldState(
            raw={
                "x": 0,
                "y": 0,
                "level": 1,
                "inventory": [{"code": "copper_dagger", "quantity": 1}],
            }
        )

    def act(self, action: Action) -> ActionResult:
        return ActionResult(ok=True, status_code=200)


def _loop(world: object, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        decoder=_ActDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=world,  # type: ignore[arg-type]
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="craft a copper dagger",
        goal_item="copper_dagger",
    )


def test_stop_on_goal(tmp_path: Path) -> None:
    turns = _loop(_GoalWorld(), tmp_path).play(10)  # dagger appears after 2 acts
    assert len(turns) == 2  # stopped early on goal (count rose 0 -> 1)


def test_stop_on_goal_ignores_residual_inventory(tmp_path: Path) -> None:
    # the dagger is present from turn 0 (count never increases) -> never "completes"
    turns = _loop(_ContaminatedWorld(), tmp_path).play(3)
    assert len(turns) == 3  # ran the full budget, no false completion


def test_empty_decode_warns_and_flags(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    loop = AgentLoop(
        decoder=_EmptyDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=kernel.load_plugin("world", "mock", {}),
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="do something",
    )
    with caplog.at_level(logging.WARNING, logger="pumpkinspice.loop"):
        turns = loop.play(1)

    assert turns[0].action["kind"] == "rest"  # nothing to parse -> rest
    assert turns[0].decoder_empty is True
    assert any("EMPTY output" in r.message for r in caplog.records)


class _RaisingDecoder:
    """Simulates a transient decoder failure (timeout / connection reset)."""

    name = "raising"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        raise RuntimeError("simulated transport failure")


def test_decoder_failure_costs_one_turn_not_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loop = AgentLoop(
        decoder=_RaisingDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=kernel.load_plugin("world", "mock", {}),
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="do something",
    )
    with caplog.at_level(logging.ERROR, logger="pumpkinspice.loop"):
        turns = loop.play(2)

    # the run SURVIVES the failing decoder: both turns recorded as empty/rest
    assert len(turns) == 2
    assert all(t.action["kind"] == "rest" and t.decoder_empty for t in turns)
    assert any("decoder call FAILED" in r.message for r in caplog.records)


class _FightWorld:
    """Returns a WON fight vs chicken ONLY for a fight action (win is action-DEPENDENT, so
    the test would catch a mis-parsed/wrong action or an over-eager stop); other actions
    return a plain ok with no fight block."""

    name = "fightworld"

    def __init__(self, result: str = "win") -> None:
        self.result = result

    def get_state(self) -> WorldState:
        return WorldState(raw={"x": 0, "y": 0, "level": 1})

    def act(self, action: Action) -> ActionResult:
        if action.kind != "fight":
            return ActionResult(ok=True, status_code=200, data={})
        return ActionResult(
            ok=True,
            status_code=200,
            data={"fight": {"result": self.result, "monster": "chicken"}},
        )


class _FightDecoder:
    name = "fight"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "fight", "args": {}}'


def _fight_loop(world: object, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        decoder=_FightDecoder(),
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=world,  # type: ignore[arg-type]
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="beat a chicken",
        goal_monster="chicken",
    )


def test_stop_on_goal_monster(tmp_path: Path) -> None:
    turns = _fight_loop(_FightWorld(result="win"), tmp_path).play(5)
    assert len(turns) == 1  # stopped after the first winning fight, not the full budget


def test_goal_monster_loss_plays_full_budget(tmp_path: Path) -> None:
    # a LOSS never satisfies goal_monster, so the run must NOT stop early (negative control
    # against an over-eager stop that would fire on any fight turn)
    turns = _fight_loop(_FightWorld(result="lose"), tmp_path).play(3)
    assert len(turns) == 3


def test_fight_won_vs() -> None:
    win = {"data": {"fight": {"result": "win", "monster": "chicken"}}}
    assert fight_won_vs(win, "chicken") is True
    assert fight_won_vs(win, "cow") is False  # won, but not the objective monster
    # "lose" is HeroBench's actual FightResult token (not "loss")
    assert (
        fight_won_vs({"data": {"fight": {"result": "lose", "monster": "chicken"}}}, "chicken")
        is False
    )
    assert fight_won_vs({"ok": True}, "chicken") is False  # no data
    assert fight_won_vs({"data": {}}, "chicken") is False  # data but no fight (non-fight action)
    assert fight_won_vs("boom", "chicken") is False  # non-dict outcome guard (foreign capture)
