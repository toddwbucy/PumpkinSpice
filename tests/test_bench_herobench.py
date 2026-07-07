"""Tests for the HeroBench planning ramp + labeler (issues #7, #8).

Offline: synthetic capture rows exercise the episode-level correctness (via
analyze) and the tier difficulty label. No model, no server.
"""

from __future__ import annotations

from typing import Any

from pumpkinspice.introspect.bench_herobench import (
    MAX_TURNS,
    RAMP,
    TASK_TYPE,
    eventual_correct,
    label_fn,
)


def _turn(**world: Any) -> dict[str, Any]:
    return {
        "index": 0,
        "task": "t",
        "world_state": world,
        "retrieval": {},
        "prompt": "p",
        "raw_output": "o",
        "action": {},
        "outcome": {"ok": True},
        "timings_ms": {},
    }


def test_ramp_integrity() -> None:
    assert set(RAMP) == {
        "control_gather",
        "chicken_level2",
        "copper_dagger",
        "yellow_slime",
        "weaponcrafting5",
    }
    # easy = doable from the starting kit; hard = needs the crafting chain
    assert RAMP["control_gather"].hard is False
    assert RAMP["chicken_level2"].hard is False
    assert all(RAMP[n].hard for n in ("copper_dagger", "yellow_slime", "weaponcrafting5"))
    assert all(t.task and t.name for t in RAMP.values())
    assert MAX_TURNS == 100


def test_label_fn_level_goal() -> None:
    reached = [_turn(level=1), _turn(level=2)]  # climbed to level 2
    lf = label_fn(reached, RAMP["chicken_level2"])
    assert lf(reached[0]) == (TASK_TYPE, True, False)

    stuck = [_turn(level=1), _turn(level=1)]  # never leveled
    assert label_fn(stuck, RAMP["chicken_level2"])(stuck[0]) == (TASK_TYPE, False, False)


def test_label_fn_item_goal_and_hard_flag() -> None:
    # ash_wood absent at start, present at end -> correct; control tier is easy
    got = [_turn(inventory=[]), _turn(inventory=[{"code": "ash_wood", "quantity": 3}])]
    assert label_fn(got, RAMP["control_gather"])(got[-1]) == (TASK_TYPE, True, False)

    # copper_dagger never crafted -> incorrect, and the tier is hard
    none = [_turn(inventory=[]), _turn(inventory=[{"code": "copper_ore", "quantity": 5}])]
    assert label_fn(none, RAMP["copper_dagger"])({}) == (TASK_TYPE, False, True)


def test_label_is_eventual_and_shared_across_turns() -> None:
    # correctness is the run's EVENTUAL outcome, so even the pre-goal first turn is
    # labeled correct once the run later reaches the goal.
    rows = [_turn(level=1), _turn(level=1), _turn(level=2)]
    lf = label_fn(rows, RAMP["chicken_level2"])
    assert all(lf(r) == (TASK_TYPE, True, False) for r in rows)


def test_eventual_correct_skill_goal() -> None:
    reached = [_turn(weaponcrafting_level=1), _turn(weaponcrafting_level=5)]
    assert eventual_correct(reached, RAMP["weaponcrafting5"]) is True
    assert eventual_correct([_turn(weaponcrafting_level=3)], RAMP["weaponcrafting5"]) is False
