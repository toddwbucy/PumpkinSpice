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
from typing import TYPE_CHECKING, Any

from pumpkinspice import analyze
from pumpkinspice.loop import fight_won_vs  # lightweight core helper (no numpy)

if TYPE_CHECKING:
    # Only needed for the LabelFn return annotation (a string under `from __future__
    # import annotations`). Importing pipeline at runtime would drag in the numpy
    # replay stack, breaking this "pure/offline" module on a core-only install.
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
    # v2 capability-milestone goal: scored on WINNING a fight vs this monster (from the
    # fight response), not a drop or a level proxy. See loop.fight_won_vs.
    goal_monster: str | None = None


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
            "Defeat Yellow Slimes and reach character level 3. A Yellow Slime resists "
            "earth damage, so first craft a weapon of a different type -- the copper "
            "dagger deals air damage.",
            hard=True,
            # Scored by character level, NOT the yellow_slimeball drop: that drop is
            # rate 20 (~5% per kill), so a genuine kill would score wrong ~95% of the
            # time. Level 3 is a reliable proxy for sustained combat past the chicken tier.
            goal_level=3,
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


def _v2_task(monster_phrase: str) -> str:
    """A fair, ICL-free capability-milestone prompt: state the goal and tell the model to
    plan FROM RETRIEVAL (no worked example, no solution spelled out) -- the model must look
    up the monster's level/resistances and decide how to gear and level up."""
    return (
        f"Defeat a {monster_phrase}. You start at level 1 with a wooden stick equipped. "
        "Check the monster's level and resistances in the reference notes and plan how to "
        "beat it -- gathering, crafting a suitable weapon, leveling up, and healing as needed."
    )


# v2 capability-milestone ladder (docs/floor-test-v2-design.md 8.1). Each task is "win a
# fight vs monster M", scored by loop.fight_won_vs -- the WIN is authoritative, unlike the
# ~5%-rate drop (which is why the RAMP's yellow_slime tier had to fall back to goal_level).
# Difficulty = level-gap from a fresh L1 char x counter-element: easy (hard=False) = winnable
# at/near L1 with the right element (chicken has no resist; yellow_slime resists earth so the
# stick is weak -> the L1 copper_dagger's air damage wins); hard (hard=True) = the air-resist
# and high-HP tiers that require leveling to out-damage. The roster caps distinct monsters per
# difficulty class (~2 easy, ~4 hard) -- the documented kill1-generalization limit.
V2_LADDER: dict[str, RampTask] = {
    t.name: t
    for t in (
        RampTask("v2_chicken", _v2_task("chicken"), hard=False, goal_monster="chicken"),
        RampTask(
            "v2_yellow_slime", _v2_task("yellow slime"), hard=False, goal_monster="yellow_slime"
        ),
        RampTask("v2_green_slime", _v2_task("green slime"), hard=True, goal_monster="green_slime"),
        RampTask("v2_blue_slime", _v2_task("blue slime"), hard=True, goal_monster="blue_slime"),
        RampTask("v2_red_slime", _v2_task("red slime"), hard=True, goal_monster="red_slime"),
        RampTask("v2_cow", _v2_task("cow"), hard=True, goal_monster="cow"),
    )
}


def eventual_correct(turns: list[dict[str, Any]], task: RampTask) -> bool:
    """Episode-level correctness: did the run reach ``task``'s goal within its budget?

    For a v2 capability goal (``goal_monster``) success = the run WON a fight vs that
    monster (loop.fight_won_vs on the captured outcomes) -- the same detector the runtime
    goal-stop uses, so runtime and post-hoc agree. Otherwise reuses analyze.analyze_turns
    (goal_item/level/skill), the harness's own success semantics."""
    if task.goal_monster is not None:
        return any(fight_won_vs(t.get("outcome", {}) or {}, task.goal_monster) for t in turns)
    metrics = analyze.analyze_turns(
        "floortest",
        turns,
        goal_item=task.goal_item,
        goal_level=task.goal_level,
        goal_skill=task.goal_skill,
    )
    return bool(metrics.success)


def make_label_fn(turns: list[dict[str, Any]], task: RampTask) -> LabelFn:
    """Build a per-turn labeler: every turn of the run gets (planning, eventual-correct,
    tier-hard). Correctness is computed once over the whole run and shared across turns,
    so it is the trajectory's EVENTUAL outcome, not its instantaneous state."""
    correct = eventual_correct(turns, task)
    hard = task.hard

    def _label(_turn: dict[str, Any]) -> tuple[str, bool, bool]:
        return (TASK_TYPE, correct, hard)

    return _label
