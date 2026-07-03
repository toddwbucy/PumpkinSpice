"""Render the HeroBench encyclopedia into belief-node texts (build-side).

Pure, deterministic rendering of HeroBench's game data (items, monsters,
resources, map locations) into natural-language "belief nodes" -- the corpus a
conventional RAG agent retrieves over. The IO (reading files), embedding, and
database upsert live in ``scripts/seed_corpus.py``; this module is kept pure so
it can be type-checked and unit-tested.

This is build-side content preparation, not runtime retrieval.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CorpusNode:
    """One belief node destined for ``kg.belief_nodes``."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _rate(rate: int) -> str:
    """HeroBench drop ``rate`` is "1 in N"."""
    return f"1/{rate}"


def render_item(d: dict[str, Any]) -> CorpusNode:
    head = f"Item: {d['name']} (code {d['code']}), level {d.get('level', '?')}, {d.get('type', '')}"
    if d.get("subtype"):
        head += f"/{d['subtype']}"
    parts = [head + "."]
    if d.get("description"):
        parts.append(str(d["description"]))
    effects = d.get("effects") or []
    if effects:
        parts.append("Effects: " + ", ".join(f"{e['name']} {e['value']}" for e in effects) + ".")
    craft = d.get("craft")
    if craft:
        ingredients = ", ".join(f"{i['quantity']}x {i['code']}" for i in craft.get("items", []))
        parts.append(
            f"Crafted with {craft.get('skill')} level {craft.get('level')} "
            f"from {ingredients} (yields {craft.get('quantity', 1)})."
        )
    return CorpusNode(
        id=f"item:{d['code']}",
        text=" ".join(parts),
        metadata={"kind": "item", "code": d["code"], "level": d.get("level")},
    )


def render_monster(d: dict[str, Any]) -> CorpusNode:
    attacks = {k[len("attack_") :]: v for k, v in d.items() if k.startswith("attack_") and v}
    resists = {k[len("res_") :]: v for k, v in d.items() if k.startswith("res_") and v}
    parts = [f"Monster: {d['name']} (code {d['code']}), level {d['level']}, {d['hp']} HP."]
    if attacks:
        parts.append("Attacks: " + ", ".join(f"{k} {v}" for k, v in attacks.items()) + ".")
    if resists:
        parts.append("Resistances: " + ", ".join(f"{k} {v}" for k, v in resists.items()) + ".")
    drops = d.get("drops") or []
    if drops:
        parts.append(
            "Drops: " + ", ".join(f"{dr['code']} ({_rate(dr['rate'])})" for dr in drops) + "."
        )
    parts.append(f"Gold {d.get('min_gold', 0)}-{d.get('max_gold', 0)}.")
    return CorpusNode(
        id=f"monster:{d['code']}",
        text=" ".join(parts),
        metadata={"kind": "monster", "code": d["code"], "level": d.get("level")},
    )


def render_resource(d: dict[str, Any]) -> CorpusNode:
    drops = ", ".join(dr["code"] for dr in d.get("drops", []))
    text = (
        f"Resource: {d['name']} (code {d['code']}), gathered with "
        f"{d['skill']} level {d['level']}. Yields: {drops}."
    )
    return CorpusNode(
        id=f"resource:{d['code']}",
        text=text,
        metadata={"kind": "resource", "code": d["code"], "level": d.get("level")},
    )


def render_locations(maps: list[dict[str, Any]], max_coords: int = 20) -> list[CorpusNode]:
    """Aggregate the per-tile map into one belief node per content code."""
    locs: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for m in maps:
        content = m.get("content")
        if not content:
            continue
        locs[(content["type"], content["code"])].append((int(m["x"]), int(m["y"])))

    nodes: list[CorpusNode] = []
    for (ctype, code), coords in locs.items():
        coords.sort()
        shown = ", ".join(f"({x},{y})" for x, y in coords[:max_coords])
        more = "" if len(coords) <= max_coords else f" (+{len(coords) - max_coords} more)"
        nodes.append(
            CorpusNode(
                id=f"location:{ctype}:{code}",
                text=(
                    f"Location: {ctype} '{code}' is found at {len(coords)} tile(s): {shown}{more}."
                ),
                metadata={
                    "kind": "location",
                    "content_type": ctype,
                    "code": code,
                    "count": len(coords),
                },
            )
        )
    return nodes


def render_mechanics() -> list[CorpusNode]:
    """Worked-example mechanic nodes: HOW skills level up, not just item/recipe
    FACTS. HeroBench's Data/*.json (and so the rest of this corpus) only encodes
    craft *requirements* -- the leveling mechanic itself is discoverable only from
    live server responses (a craft's ``details.xp``), which a conventional agent
    only sees AFTER choosing to craft. A real practitioner's knowledge base would
    document the mechanic explicitly, same as a game wiki does; these nodes fill
    that gap with a worked example (copper_dagger, the cheapest weaponcrafting
    recipe) generalized to any skill/recipe pair."""
    return [
        CorpusNode(
            id="mechanic:crafting_xp",
            text=(
                "Mechanic: crafting an item grants XP in the skill named in its recipe "
                "(e.g. crafting copper_dagger, recipe skill weaponcrafting, grants "
                "weaponcrafting XP), regardless of whether you needed the crafted item. "
                "Enough XP raises that skill's level. Worked example: crafting ONE "
                "copper_dagger (6x copper, at the weaponcrafting workshop, needs only "
                "weaponcrafting level 1) grants about 150 weaponcrafting XP -- enough to "
                "raise weaponcrafting from level 1 to level 2 in a single craft."
            ),
            metadata={"kind": "mechanic", "code": "crafting_xp"},
        ),
        CorpusNode(
            id="mechanic:leveling_via_crafting",
            text=(
                "Mechanic: to reach a skill level required by a HIGH-level recipe, "
                "repeatedly craft any LOW-level recipe in that same skill -- each craft "
                "grants XP toward the skill even if you discard or do not need the item. "
                "Worked example: sticky_sword needs weaponcrafting level 5. To reach "
                "weaponcrafting level 5, repeatedly gather copper_ore, smelt it into "
                "copper, and craft copper_dagger (weaponcrafting level 1, 6x copper) at "
                "the weaponcrafting workshop until weaponcrafting reaches level 5, THEN "
                "gather sticky_sword's own ingredients and craft it. The same pattern "
                "applies to any skill: level it on its cheapest recipe before attempting "
                "a recipe that needs a higher level."
            ),
            metadata={"kind": "mechanic", "code": "leveling_via_crafting"},
        ),
        CorpusNode(
            id="mechanic:combat_risk",
            text=(
                "Mechanic: fighting a monster can be WON or LOST depending on the "
                "character's level and equipped weapon versus the monster's level and "
                "attacks. A lost fight yields 0 XP and 0 drops and can reduce HP to 0 "
                "(defeat, requiring rest before fighting again) -- a lost fight is still "
                "an HTTP success, so 'the action succeeded' does not mean 'the fight was "
                "won'. Check a monster's level against your own before engaging; a low-"
                "level character will lose repeatedly to a monster several levels above it."
            ),
            metadata={"kind": "mechanic", "code": "combat_risk"},
        ),
    ]


def load_corpus(data_dir: Path) -> list[CorpusNode]:
    """Load HeroBench's Data/*.json and render the full belief-node corpus."""

    def read(name: str) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = json.loads((data_dir / name).read_text())
        return data

    nodes: list[CorpusNode] = []
    nodes += [render_item(d) for d in read("items.json")]
    nodes += [render_monster(d) for d in read("monsters.json")]
    nodes += [render_resource(d) for d in read("resources.json")]
    nodes += render_locations(read("maps.json"))
    nodes += render_mechanics()
    return nodes
