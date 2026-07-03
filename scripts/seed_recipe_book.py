#!/usr/bin/env python
"""Build-side seeder: a flattened "recipe book" for every craftable item.

The relational tables (kg.item_craft / craft_ingredients / sources / locations)
hold the recipe GRAPH one edge at a time, so answering "how do I make X from
scratch?" needs a multi-hop traversal -- which a one-hop runtime join cannot do
(it dead-ends at a *crafted* ingredient like `copper`, never reaching the ore or
its location). This script does that traversal ONCE, build-side, and writes the
fully-flattened, rolled-up plan into kg.recipe_book so the runtime retrieval is a
plain lookup. Complexity lives here (where we can audit it), not in the hot path.

Each row: item_code -> readable recipe_text + structured steps/raw_materials.
The traversal is conventional graph walking over our own tables; it is NOT HADES
structural retrieval and the model never runs it -- the agent only ever reads the
finished book via plain top-k + a lookup.

Run as the READ-WRITE loader role (the loader owns kg, so the ALTER DEFAULT
PRIVILEGES from bootstrap_pg.py auto-grant SELECT to the read-only agent). Idempotent.

  uv run --extra pgvector python scripts/seed_recipe_book.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field

import psycopg


@dataclass
class Graph:
    craft: dict[str, tuple[str, int, int]]  # item -> (skill, craft_level, yield_qty)
    ingredients: dict[str, list[tuple[str, int]]]  # item -> [(ingredient, qty), ...]
    sources: dict[str, tuple[str, str]]  # item -> (source_type, source_code)
    loc: dict[str, tuple[int, int]]  # code -> (x, y)
    items: dict[str, tuple[str, int, str]]  # code -> (name, level, type)


def load_graph(cur: psycopg.Cursor) -> Graph:
    cur.execute("SELECT item_code, skill, craft_level, yield_qty FROM kg.item_craft")
    craft = {r[0]: (r[1], int(r[2] or 1), int(r[3] or 1)) for r in cur.fetchall()}
    cur.execute("SELECT item_code, ingredient_code, quantity FROM kg.craft_ingredients")
    ingredients: dict[str, list[tuple[str, int]]] = {}
    for item, ing, qty in cur.fetchall():
        ingredients.setdefault(item, []).append((ing, int(qty)))
    cur.execute("SELECT item_code, source_type, source_code FROM kg.sources")
    sources = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.execute("SELECT code, x, y FROM kg.locations")
    loc = {r[0]: (int(r[1]), int(r[2])) for r in cur.fetchall()}
    cur.execute("SELECT code, name, level, type FROM kg.items")
    items = {r[0]: (r[1], int(r[2] or 0), r[3]) for r in cur.fetchall()}
    return Graph(craft, ingredients, sources, loc, items)


@dataclass
class Resolution:
    craft_steps: list[dict] = field(default_factory=list)  # post-order: intermediates -> final
    raw: dict[str, int] = field(default_factory=dict)  # gatherable -> total quantity
    unresolved: list[str] = field(default_factory=list)  # no recipe AND no source


def _loc_str(g: Graph, code: str) -> str:
    xy = g.loc.get(code)
    return f"{code} ({xy[0]},{xy[1]})" if xy else f"{code} (location unknown)"


def _xy(g: Graph, code: str) -> str:
    xy = g.loc.get(code)
    return f"({xy[0]},{xy[1]})" if xy else "(location unknown)"


def resolve(item: str, qty: int, g: Graph, res: Resolution, seen: frozenset[str]) -> None:
    """Walk the dependency tree, accumulating raw-material totals and an ordered
    (post-order) list of craft steps. Cycle-guarded via `seen`."""
    if item in seen:  # defensive: recipe graphs should be acyclic
        return
    craft = g.craft.get(item)
    if craft is None:
        # Gatherable/dropped raw material (or a base item the agent may start with).
        if item in g.sources:
            res.raw[item] = res.raw.get(item, 0) + qty
        elif item not in res.raw:
            res.unresolved.append(item)
        return
    skill, craft_level, yield_qty = craft
    ops = math.ceil(qty / max(yield_qty, 1))
    for ing, q in g.ingredients.get(item, []):
        resolve(ing, q * ops, g, res, seen | {item})
    res.craft_steps.append(
        {"item": item, "qty": qty, "skill": skill, "level": craft_level, "workshop": skill}
    )


def build_recipe(item: str, g: Graph) -> dict | None:
    if item not in g.craft:
        return None
    res = Resolution()
    resolve(item, 1, g, res, frozenset())

    name, level, itype = g.items.get(item, (item, 0, "?"))
    lines = [f"Recipe for {item} ({name}, level {level} {itype}):"]
    # Ordered plan: gather raw materials, then craft bottom-up (intermediates -> final).
    steps: list[str] = []
    n = 1
    for raw_code, raw_qty in sorted(res.raw.items()):
        stype, scode = g.sources.get(raw_code, ("resource", raw_code))
        verb = "Fight" if stype == "monster" else "Gather"
        steps.append(f"  {n}. {verb} {raw_qty}x {raw_code} at {_loc_str(g, scode)}.")
        n += 1
    for st in res.craft_steps:
        steps.append(
            f"  {n}. Craft {st['qty']}x {st['item']} at the {st['skill']} workshop "
            f"{_xy(g, st['workshop'])} (requires {st['skill']} level {st['level']})."
        )
        n += 1
    # Skill levels required across the whole chain (max per skill).
    skills: dict[str, int] = {}
    for st in res.craft_steps:
        skills[st["skill"]] = max(skills.get(st["skill"], 0), int(st["level"]))
    if skills:
        lines.append(
            "Skills required: "
            + ", ".join(f"{s} level {lvl}" for s, lvl in sorted(skills.items()))
            + "."
        )
    lines.append("Ordered steps (do gathers first, then craft bottom-up):")
    lines.extend(steps)
    if res.raw:
        lines.append(
            "Raw materials to gather: "
            + ", ".join(f"{q}x {c}" for c, q in sorted(res.raw.items()))
            + "."
        )
    if res.unresolved:
        lines.append("Note: no known source for: " + ", ".join(sorted(set(res.unresolved))) + ".")
    return {
        "item_code": item,
        "recipe_text": "\n".join(lines),
        "steps": json.dumps(res.craft_steps),
        "raw_materials": json.dumps(res.raw),
    }


DDL = """
CREATE TABLE IF NOT EXISTS kg.recipe_book (
    item_code     text PRIMARY KEY,
    recipe_text   text NOT NULL,
    steps         jsonb,
    raw_materials jsonb
)
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn-env", default="PUMPKINSPICE_PG_LOADER_DSN")
    args = ap.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(
            f"error: ${args.dsn_env} not set (run scripts/bootstrap_pg.py first).", file=sys.stderr
        )
        return 1

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        g = load_graph(cur)
        rows = [r for item in g.craft if (r := build_recipe(item, g)) is not None]
        cur.execute("TRUNCATE kg.recipe_book")
        cur.executemany(
            "INSERT INTO kg.recipe_book (item_code, recipe_text, steps, raw_materials) "
            "VALUES (%(item_code)s, %(recipe_text)s, %(steps)s, %(raw_materials)s)",
            rows,
        )
        conn.commit()
        print(f"seeded kg.recipe_book: {len(rows)} craftable items")
        sample = next(
            (r for r in rows if r["item_code"] == "copper_dagger"), rows[0] if rows else None
        )
        if sample:
            print("\n--- sample: copper_dagger ---\n" + sample["recipe_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
