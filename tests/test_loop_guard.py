"""The empty-content guard: an empty decode warns and is flagged in the capture."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from pumpkinspice import kernel
from pumpkinspice.contracts import Action, ActionResult, WorldState
from pumpkinspice.loop import AgentLoop


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
