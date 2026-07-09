"""Tests for the HeroBench planning ramp + labeler (issues #7, #8).

Offline: synthetic capture rows exercise the episode-level correctness (via
analyze) and the tier difficulty label. No model, no server.
"""

from __future__ import annotations

from typing import Any

import pytest

# bench_herobench itself only needs `analyze` (core), but it lives in the introspect
# package whose __init__ re-exports geometry (numpy), so importing it needs the extra.
pytest.importorskip("numpy")

from pumpkinspice.introspect.bench_herobench import (
    MAX_TURNS,
    RAMP,
    TASK_TYPE,
    V2_LADDER,
    eventual_correct,
    make_label_fn,
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


def _fight_turn(result: str, monster: str) -> dict[str, Any]:
    """A turn whose captured outcome is a HeroBench fight response (win/loss vs a monster)."""
    t = _turn()
    t["action"] = {"kind": "fight", "args": {}}
    t["outcome"] = {"ok": True, "data": {"fight": {"result": result, "monster": monster}}}
    return t


def test_v2_ladder_integrity() -> None:
    assert set(V2_LADDER) == {
        "v2_chicken",
        "v2_yellow_slime",
        "v2_green_slime",
        "v2_blue_slime",
        "v2_red_slime",
        "v2_cow",
    }
    # every v2 task is scored on winning a fight vs its monster (not an item/level proxy)
    assert all(
        t.goal_monster and t.goal_item is None and t.goal_level is None for t in V2_LADDER.values()
    )
    assert V2_LADDER["v2_cow"].goal_monster == "cow"
    # easy = winnable at/near L1 (no resist, or earth-resist beaten by the L1 air dagger);
    # hard = the air-resist / high-HP tiers that require leveling
    assert V2_LADDER["v2_chicken"].hard is False
    assert V2_LADDER["v2_yellow_slime"].hard is False
    assert all(
        V2_LADDER[n].hard for n in ("v2_green_slime", "v2_blue_slime", "v2_red_slime", "v2_cow")
    )
    assert all(t.task and t.name for t in V2_LADDER.values())


def test_eventual_correct_goal_monster_scores_on_win() -> None:
    task = V2_LADDER["v2_yellow_slime"]
    # a win vs the objective monster -> correct (even with earlier non-fight turns)
    assert eventual_correct([_turn(level=1), _fight_turn("win", "yellow_slime")], task) is True
    # a LOSS vs the objective monster does not count
    assert eventual_correct([_fight_turn("loss", "yellow_slime")], task) is False
    # a WIN vs a DIFFERENT monster does not count (grades the objective monster only)
    assert eventual_correct([_fight_turn("win", "chicken")], task) is False
    # no fights at all -> not correct
    assert eventual_correct([_turn(level=1)], task) is False


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
    lf = make_label_fn(reached, RAMP["chicken_level2"])
    assert lf(reached[0]) == (TASK_TYPE, True, False)

    stuck = [_turn(level=1), _turn(level=1)]  # never leveled
    assert make_label_fn(stuck, RAMP["chicken_level2"])(stuck[0]) == (TASK_TYPE, False, False)


def test_label_fn_item_goal_and_hard_flag() -> None:
    # ash_wood absent at start, present at end -> correct; control tier is easy
    got = [_turn(inventory=[]), _turn(inventory=[{"code": "ash_wood", "quantity": 3}])]
    assert make_label_fn(got, RAMP["control_gather"])(got[-1]) == (TASK_TYPE, True, False)

    # copper_dagger never crafted -> incorrect, and the tier is hard
    none = [_turn(inventory=[]), _turn(inventory=[{"code": "copper_ore", "quantity": 5}])]
    assert make_label_fn(none, RAMP["copper_dagger"])({}) == (TASK_TYPE, False, True)


def test_label_is_eventual_and_shared_across_turns() -> None:
    # correctness is the run's EVENTUAL outcome, so even the pre-goal first turn is
    # labeled correct once the run later reaches the goal.
    rows = [_turn(level=1), _turn(level=1), _turn(level=2)]
    lf = make_label_fn(rows, RAMP["chicken_level2"])
    assert all(lf(r) == (TASK_TYPE, True, False) for r in rows)


def test_eventual_correct_skill_goal() -> None:
    reached = [_turn(weaponcrafting_level=1), _turn(weaponcrafting_level=5)]
    assert eventual_correct(reached, RAMP["weaponcrafting5"]) is True
    assert eventual_correct([_turn(weaponcrafting_level=3)], RAMP["weaponcrafting5"]) is False


def test_configs_match_ramp() -> None:
    # Enforce the "can't drift" claim: each config's task + budget mirror RAMP, and
    # none sets goal_item (the run must go the full budget; scoring is done afterward).
    import tomllib
    from pathlib import Path

    for name, task in RAMP.items():
        cfg = tomllib.loads(Path(f"configs/floor_planning_{name}.toml").read_text())
        assert cfg["run"]["task"] == task.task, f"{name}: config task drifted from RAMP"
        assert cfg["run"]["max_turns"] == MAX_TURNS
        assert "goal_item" not in cfg["run"], f"{name}: config must not stop early on a goal"
