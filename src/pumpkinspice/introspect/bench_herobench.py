"""Calibrated HeroBench planning ramp + labeler for the #7/#8 floor test.

HeroBench is a crafting-gated, type-matching PLANNING benchmark (not a leveling
ladder): every slime resists its OWN attack type, so beating a monster above your
level means planning the gather -> smelt/craft -> equip counter-gear chain. Example:
a Yellow Slime resists earth, so a fresh character's earth-only wooden stick is
blunted -- but crafting the air-damage ``copper_dagger`` (weaponcrafting 1, mine 48
copper_ore -> smelt 6 copper -> craft) beats it at level 1. Difficulty is the DEPTH of
that dependency chain, not raw level.

The ramp gives every task a fixed 100-turn budget with NO stop-on-goal, so we observe
how far a model climbs. Correctness is scored AFTER the run, episode-level, via
``analyze.analyze_turns``: did the character reach the task's goal within the budget.
All turns of a run share that one label -- matching #7's "eventual-correct vs
eventual-wrong" framing. The tier's ``hard`` flag is the INDEPENDENT difficulty label
for #7's kill #1: easy = doable with the starting kit (no crafting), hard = requires
the gear-dependency chain.

Retrieval must be the strongest arm (the corpus exposes monster resistances AND weapon
damage types, so the type-matching is a real planning signal, not retrieval
starvation). This module is pure/offline: it labels recorded captures; it never runs a
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pumpkinspice import analyze
from pumpkinspice.introspect.pipeline import LabelFn

# The task type these turns carry into the evaluator (contrasts with MATH "reasoning").
TASK_TYPE = "planning"
# Fixed budget: run this many turns and see how far the model climbs (no early stop).
MAX_TURNS = 100


@dataclass(frozen=True)
class RampTask:
    """One tier of the calibrated ramp: the task string, its independent difficulty
    label, and the goal used to score episode-level correctness afterward."""

    name: str
    task: str
    hard: bool  # independent difficulty label (kill #1): requires the crafting chain?
    goal_item: str | None = None
    goal_level: int | None = None
    goal_skill: str | None = None


# Difficulty rises with gear-dependency depth. easy (hard=False) = doable from the
# starting kit; hard (hard=True) = requires mining/smelting/crafting the counter-gear.
RAMP: dict[str, RampTask] = {
    t.name: t
    for t in (
        RampTask(
            "control_gather",
            "Gather ash wood: move to an ash tree and gather it.",
            hard=False,
            goal_item="ash_wood",
        ),
        RampTask(
            "chicken_level2",
            "Reach character level 2 by fighting chickens. A chicken is a safe "
            "level-1 fight; each win grants 50 XP, and 150 XP reaches level 2.",
            hard=False,
            goal_level=2,
        ),
        RampTask(
            "copper_dagger",
            "Craft a copper dagger. Mine copper ore, smelt it into copper, then "
            "craft the dagger (weaponcrafting level 1).",
            hard=True,
            goal_item="copper_dagger",
        ),
        RampTask(
            "yellow_slime",
            "Defeat a Yellow Slime. It resists earth damage, so first craft a weapon "
            "of a different damage type -- the copper dagger deals air damage.",
            hard=True,
            goal_item="yellow_slimeball",
        ),
        RampTask(
            "weaponcrafting5",
            "Reach weaponcrafting level 5 (the prerequisite for tier-5 weapons such "
            "as the sticky sword).",
            hard=True,
            goal_skill="weaponcrafting",
            goal_level=5,
        ),
    )
}


def eventual_correct(turns: list[dict[str, Any]], task: RampTask) -> bool:
    """Episode-level correctness: did the run reach ``task``'s goal within its budget?
    Reuses analyze.analyze_turns so this matches the harness's own success semantics."""
    metrics = analyze.analyze_turns(
        "floortest",
        turns,
        goal_item=task.goal_item,
        goal_level=task.goal_level,
        goal_skill=task.goal_skill,
    )
    return bool(metrics.success)


def label_fn(turns: list[dict[str, Any]], task: RampTask) -> LabelFn:
    """Build a per-turn labeler: every turn of the run gets (planning, eventual-correct,
    tier-hard). Correctness is computed once over the whole run and shared across turns,
    so it is the trajectory's EVENTUAL outcome, not its instantaneous state."""
    correct = eventual_correct(turns, task)
    hard = task.hard

    def _label(_turn: dict[str, Any]) -> tuple[str, bool, bool]:
        return (TASK_TYPE, correct, hard)

    return _label
