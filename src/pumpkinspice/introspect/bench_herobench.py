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
    # Independent difficulty label for kill #1. Its SEMANTICS are per-ladder, and RAMP vs
    # V2_LADDER captures are SEPARATE corpora that must not be pooled on this axis: in RAMP
    # hard = requires the crafting chain; in V2_LADDER hard = requires LEVELING to out-damage
    # (yellow_slime needs a craft but no leveling, so it is hard=True in RAMP, hard=False in
    # V2_LADDER -- the same monster, two different difficulty axes).
    hard: bool
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
    """A fair, ICL-free capability-milestone prompt: state the goal and point the model at
    retrieval to plan. It does NOT name the solution steps (gather/craft/level/heal) -- that
    is the planning the IV measures, and naming it would bias the arm and push the chicken
    positive control into needless crafting."""
    return (
        f"Defeat a {monster_phrase}. You start at level 1 with a wooden stick equipped. "
        "Use the reference notes to plan how to beat it."
    )


# v2 capability-milestone ladder (docs/floor-test-v2-design.md 8.1). Each task is "win a
# fight vs monster M", scored by loop.fight_won_vs -- the WIN is authoritative, unlike the
# ~5%-rate drop (which is why the RAMP's yellow_slime tier had to fall back to goal_level).
# Difficulty = level-gap from a fresh L1 char x counter-element (this ladder's `hard` axis,
# DISTINCT from RAMP's crafting-chain axis; see RampTask.hard): easy (hard=False) = winnable
# at/near L1 (chicken has no resist; yellow_slime resists earth but the L1 copper_dagger's air
# damage wins -- a craft, no leveling); hard (hard=True) = the air-resist and high-HP tiers
# that require LEVELING to out-damage. The roster caps distinct monsters per difficulty class
# (~2 easy, ~4 hard) -- the documented kill1-generalization limit.
#
# UNLIKE the fixed-100-turn RAMP, v2 tiers are STOP-ON-GOAL: an episode ends at the first win
# (goal_monster in [run]), and the measured metric is the FIRST planning turn (design doc 4.2),
# not the full trajectory -- so v2 captures are NOT budget-comparable to RAMP captures.
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

# One registry so CLI tier lookups / help find both ladders (RAMP is v1, V2_LADDER is v2).
TASKS: dict[str, RampTask] = {**RAMP, **V2_LADDER}


def eventual_correct(turns: list[dict[str, Any]], task: RampTask) -> bool:
    """Episode-level correctness: did the run reach ``task``'s goal within its budget?

    Delegates entirely to analyze.analyze_turns (the single success-semantics source, incl.
    the v2 ``goal_monster`` won-fight arm), so this bench path and the analyze/sweep/web
    surfaces cannot disagree."""
    metrics = analyze.analyze_turns(
        "floortest",
        turns,
        goal_item=task.goal_item,
        goal_level=task.goal_level,
        goal_skill=task.goal_skill,
        goal_monster=task.goal_monster,
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
