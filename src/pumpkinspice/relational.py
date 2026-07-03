"""Normalize the HeroBench encyclopedia into relational rows (build-side).

HeroBench's data already has a schema: items have crafting recipes that reference
other items, monsters and resources have drop tables referencing items, and map
tiles reference entities by code. This module flattens that into normalized
tables so the pgvector retrieval arm can JOIN structured neighbours onto the
semantic hits (recipe -> ingredients -> sources -> locations).

This is plain relational structure over an explicit schema -- conventional RAG
enrichment, NOT the HADES structural-embedding machinery. Pure and tested; the
IO/DDL/insert lives in scripts/seed_relational_pg.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RelationalRows:
    items: list[dict[str, Any]] = field(default_factory=list)
    item_craft: list[dict[str, Any]] = field(default_factory=list)
    craft_ingredients: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)


def build_relational(
    items: list[dict[str, Any]],
    monsters: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    maps: list[dict[str, Any]],
) -> RelationalRows:
    rows = RelationalRows()

    for it in items:
        rows.items.append(
            {
                "code": it["code"],
                "name": it.get("name"),
                "level": it.get("level"),
                "type": it.get("type"),
                "subtype": it.get("subtype"),
            }
        )
        craft = it.get("craft")
        if craft:
            rows.item_craft.append(
                {
                    "item_code": it["code"],
                    "skill": craft.get("skill"),
                    "craft_level": craft.get("level"),
                    "yield_qty": craft.get("quantity", 1),
                }
            )
            for ing in craft.get("items", []):
                rows.craft_ingredients.append(
                    {
                        "item_code": it["code"],
                        "ingredient_code": ing["code"],
                        "quantity": ing.get("quantity", 1),
                    }
                )

    for mon in monsters:
        for drop in mon.get("drops", []):
            rows.sources.append(
                {
                    "item_code": drop["code"],
                    "source_type": "monster",
                    "source_code": mon["code"],
                    "rate": drop.get("rate"),
                }
            )

    for res in resources:
        for drop in res.get("drops", []):
            rows.sources.append(
                {
                    "item_code": drop["code"],
                    "source_type": "resource",
                    "source_code": res["code"],
                    "rate": drop.get("rate"),
                }
            )

    for m in maps:
        content = m.get("content")
        if not content:
            continue
        rows.locations.append(
            {
                "content_type": content["type"],
                "code": content["code"],
                "x": int(m["x"]),
                "y": int(m["y"]),
            }
        )

    return rows


def load_relational(data_dir: Path) -> RelationalRows:
    def read(name: str) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = json.loads((data_dir / name).read_text())
        return data

    return build_relational(
        read("items.json"),
        read("monsters.json"),
        read("resources.json"),
        read("maps.json"),
    )
