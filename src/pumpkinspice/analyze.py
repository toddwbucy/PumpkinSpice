"""Per-run metrics over captures + a cross-model comparison (Stage 1).

The planning ablation runs the same task across models and stages; this turns the
per-turn JSONL captures into comparable outcome metrics. Stage 1 (reactive loop)
measures outcome efficiency -- success, steps-to-completion, failed/no-op actions,
wasted moves, progress. Stages 2-3 add plan-change/replan metrics on top.

Pure functions over the parsed capture; the CLI (`pumpkinspice analyze`) does IO.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pumpkinspice.loop import fight_won_vs


@dataclass
class RunMetrics:
    name: str
    model: str
    backend: str
    steps: int
    success: bool | None
    failed_actions: int
    no_ops: int
    unique_positions: int
    revisits: int
    level_delta: int | None
    xp_delta: int | None
    avg_decode_ms: float
    avg_retrieval_ms: float
    # Stage 3: how many turns rewrote the committed plan (0 for reactive/Stage-2).
    replans: int = 0
    # Token throughput: makes decode latency interpretable (big prompt? long
    # generation? slow hardware?). avg_gen_tokens = mean completion tokens/turn;
    # decode_tok_s = total completion tokens / total decode seconds.
    avg_gen_tokens: float = 0.0
    decode_tok_s: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    final_inventory: dict[str, int] = field(default_factory=dict)


def _int(state: dict[str, Any], key: str) -> int | None:
    v = state.get(key)
    return int(v) if isinstance(v, int | float) else None


def _inventory(state: dict[str, Any]) -> dict[str, int]:
    inv = state.get("inventory")
    out: dict[str, int] = {}
    if isinstance(inv, dict):
        for code, qty in inv.items():
            if isinstance(qty, int | float):
                out[str(code)] = int(qty)
    elif isinstance(inv, list):
        for item in inv:
            if isinstance(item, dict) and item.get("code"):
                out[str(item["code"])] = int(item.get("quantity", 1) or 0)
    return out


def _state_after(turn: dict[str, Any]) -> dict[str, Any]:
    """The character state after a turn: the action result if it carries one, else
    the state observed at the turn's start. Craft/gather/fight responses nest the
    updated character under "character" (a SkillResponse), so unwrap that first --
    otherwise the post-craft inventory (the new dagger) is invisible to success."""
    outcome = turn.get("outcome")
    data = outcome.get("data") if isinstance(outcome, dict) else None
    if isinstance(data, dict):
        char = data.get("character")
        if isinstance(char, dict) and ("level" in char or "inventory" in char or "x" in char):
            return char
        # "solved" covers a World that self-reports goal state flatly (HanoiWorld).
        if "level" in data or "x" in data or "solved" in data:
            return data
    ws = turn.get("world_state")
    return ws if isinstance(ws, dict) else {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_turns(
    name: str,
    turns: list[dict[str, Any]],
    *,
    goal_item: str | None = None,
    goal_level: int | None = None,
    goal_skill: str | None = None,
    goal_state_key: str | None = None,
    goal_monster: str | None = None,
) -> RunMetrics:
    if not turns:
        return RunMetrics(name, "", "", 0, None, 0, 0, 0, 0, None, None, 0.0, 0.0)

    first_state = turns[0].get("world_state", {})
    final_state = _state_after(turns[-1])

    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    revisits = 0
    for t in turns:
        ws = t.get("world_state", {})
        x, y = _int(ws, "x"), _int(ws, "y")
        if x is not None and y is not None:
            pos = (x, y)
            if pos in seen:
                revisits += 1
            seen.add(pos)
            positions.append(pos)

    level_start, level_end = _int(first_state, "level"), _int(final_state, "level")
    xp_start, xp_end = _int(first_state, "xp"), _int(final_state, "xp")
    final_inv = _inventory(final_state)

    success: bool | None = None
    if goal_monster is not None:
        # v2 capability goal: won a fight vs the monster (same detector the runtime goal-stop
        # uses -> analyze and eventual_correct agree). Precedence matches loop._goal_reached.
        success = any(fight_won_vs(t.get("outcome", {}), goal_monster) for t in turns)
    elif goal_state_key is not None:
        success = bool(final_state.get(goal_state_key))
    elif goal_item is not None:
        # Crafted THIS run: more of the goal item than at the start (a reset character
        # can carry residual inventory; "present" alone is a false positive).
        first_inv = _inventory(first_state)
        success = final_inv.get(goal_item, 0) > first_inv.get(goal_item, 0)
    elif goal_level is not None:
        # goal_skill scopes the level goal to a skill (weaponcrafting_level, ...);
        # bare goal_level keeps meaning the character's combat level.
        goal_end = _int(final_state, f"{goal_skill}_level") if goal_skill else level_end
        success = goal_end is not None and goal_end >= goal_level

    gen_tokens = [int(t.get("completion_tokens", 0) or 0) for t in turns]
    decode_s = sum(t.get("timings_ms", {}).get("decode", 0.0) for t in turns) / 1000.0
    decode_tok_s = (sum(gen_tokens) / decode_s) if decode_s > 0 else 0.0

    # Replans: turns where the committed plan changed (Stage 3). The first non-empty
    # plan is the initial plan, not a replan.
    replans = 0
    prev_plan = ""
    for t in turns:
        p = str(t.get("plan", "") or "")
        if p and prev_plan and p != prev_plan:
            replans += 1
        if p:
            prev_plan = p

    return RunMetrics(
        name=name,
        model=str(turns[0].get("model") or "?"),
        backend=str(turns[0].get("retrieval", {}).get("backend") or "?"),
        steps=len(turns),
        success=success,
        failed_actions=sum(1 for t in turns if not t.get("outcome", {}).get("ok", True)),
        no_ops=sum(1 for t in turns if t.get("decoder_empty")),
        unique_positions=len(seen),
        revisits=revisits,
        level_delta=(level_end - level_start)
        if level_start is not None and level_end is not None
        else None,
        xp_delta=(xp_end - xp_start) if xp_start is not None and xp_end is not None else None,
        avg_decode_ms=_mean(
            [t["timings_ms"]["decode"] for t in turns if "decode" in t.get("timings_ms", {})]
        ),
        avg_retrieval_ms=_mean(
            [t["timings_ms"]["retrieval"] for t in turns if "retrieval" in t.get("timings_ms", {})]
        ),
        replans=replans,
        avg_gen_tokens=_mean([float(g) for g in gen_tokens]),
        decode_tok_s=decode_tok_s,
        action_counts=dict(Counter(t.get("action", {}).get("kind", "?") for t in turns)),
        final_inventory=final_inv,
    )


def load_metrics(
    path: Path,
    *,
    goal_item: str | None = None,
    goal_level: int | None = None,
    goal_skill: str | None = None,
    goal_state_key: str | None = None,
    goal_monster: str | None = None,
) -> RunMetrics:
    turns = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return analyze_turns(
        path.name,
        turns,
        goal_item=goal_item,
        goal_level=goal_level,
        goal_skill=goal_skill,
        goal_state_key=goal_state_key,
        goal_monster=goal_monster,
    )


def comparison_table(metrics: list[RunMetrics]) -> str:
    """A compact text table, sorted by success then fewest steps. The leading `run`
    column (the capture name) distinguishes same-model runs (e.g. comparing prompt
    strategies on one model); `model` distinguishes a cross-model sweep."""
    rows = sorted(metrics, key=lambda m: (m.success is not True, m.steps))
    header = (
        f"{'run':22s} {'model':16s} {'backend':14s} {'steps':>5s} {'ok':>4s} {'fail':>4s} "
        f"{'noop':>4s} {'lvlΔ':>4s} {'xpΔ':>6s} {'revis':>5s} {'rplan':>5s} {'dec_ms':>7s} "
        f"{'gen_tk':>6s} {'tok/s':>6s}"
    )
    lines = [header, "-" * len(header)]
    for m in rows:
        ok = "?" if m.success is None else ("yes" if m.success else "no")
        run = m.name[:-6] if m.name.endswith(".jsonl") else m.name
        lines.append(
            f"{run[:22]:22s} {m.model[:16]:16s} {m.backend[:14]:14s} {m.steps:5d} {ok:>4s} "
            f"{m.failed_actions:4d} {m.no_ops:4d} "
            f"{('' if m.level_delta is None else f'+{m.level_delta}'):>4s} "
            f"{('' if m.xp_delta is None else f'+{m.xp_delta}'):>6s} "
            f"{m.revisits:5d} {m.replans:5d} {m.avg_decode_ms:7.0f} "
            f"{m.avg_gen_tokens:6.0f} {m.decode_tok_s:6.1f}"
        )
    return "\n".join(lines)


def metrics_as_dicts(metrics: list[RunMetrics]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]
