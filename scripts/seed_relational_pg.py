#!/usr/bin/env python
"""Build-side seeder: normalize the HeroBench schema into kg.* relational tables.

Creates and fills the relational tables the pgvector "relational" mode joins
against: kg.items, kg.item_craft, kg.craft_ingredients, kg.sources, kg.locations.
Run as the READ-WRITE loader role ($PUMPKINSPICE_PG_LOADER_DSN). The loader owns
the kg schema, so the ALTER DEFAULT PRIVILEGES set in bootstrap_pg.py auto-grants
SELECT on these new tables to the read-only agent role. Idempotent (truncate +
insert).

Run:
  uv run --extra pgvector python scripts/seed_relational_pg.py \
      --data-dir ~/git/HeroBench/Virtual_Environment/Data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pumpkinspice.relational import RelationalRows, load_relational

DEFAULT_DATA_DIR = Path.home() / "git/HeroBench/Virtual_Environment/Data"

# (table, ordered columns, DDL) -- column order must match the row dict keys used below.
TABLES: dict[str, tuple[list[str], str]] = {
    "items": (
        ["code", "name", "level", "type", "subtype"],
        "CREATE TABLE IF NOT EXISTS kg.items "
        "(code text PRIMARY KEY, name text, level int, type text, subtype text)",
    ),
    "item_craft": (
        ["item_code", "skill", "craft_level", "yield_qty"],
        "CREATE TABLE IF NOT EXISTS kg.item_craft "
        "(item_code text PRIMARY KEY, skill text, craft_level int, yield_qty int)",
    ),
    "craft_ingredients": (
        ["item_code", "ingredient_code", "quantity"],
        "CREATE TABLE IF NOT EXISTS kg.craft_ingredients "
        "(item_code text, ingredient_code text, quantity int)",
    ),
    "sources": (
        ["item_code", "source_type", "source_code", "rate"],
        "CREATE TABLE IF NOT EXISTS kg.sources "
        "(item_code text, source_type text, source_code text, rate int)",
    ),
    "locations": (
        ["content_type", "code", "x", "y"],
        "CREATE TABLE IF NOT EXISTS kg.locations (content_type text, code text, x int, y int)",
    ),
}
INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_craft_ing_item ON kg.craft_ingredients(item_code)",
    "CREATE INDEX IF NOT EXISTS ix_sources_item ON kg.sources(item_code)",
    "CREATE INDEX IF NOT EXISTS ix_locations_code ON kg.locations(code)",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--dsn-env", default="PUMPKINSPICE_PG_LOADER_DSN")
    args = ap.parse_args(argv)

    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(
            f"error: ${args.dsn_env} not set (run scripts/bootstrap_pg.py first).", file=sys.stderr
        )
        return 1
    if not args.data_dir.exists():
        print(f"error: data dir not found: {args.data_dir}", file=sys.stderr)
        return 1

    rows: RelationalRows = load_relational(args.data_dir)
    data = {
        "items": rows.items,
        "item_craft": rows.item_craft,
        "craft_ingredients": rows.craft_ingredients,
        "sources": rows.sources,
        "locations": rows.locations,
    }

    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for table, (cols, ddl) in TABLES.items():
            cur.execute(ddl)
            cur.execute(sql.SQL("TRUNCATE kg.{}").format(sql.Identifier(table)))
            insert = sql.SQL("INSERT INTO kg.{} ({}) VALUES ({})").format(
                sql.Identifier(table),
                sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                sql.SQL(", ").join(sql.Placeholder() * len(cols)),
            )
            cur.executemany(insert, [[r.get(c) for c in cols] for r in data[table]])
            print(f"  kg.{table}: {len(data[table])} rows")
        for idx in INDEXES:
            cur.execute(idx)
    print("relational tables seeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
