"""Relational extraction is pure and deterministic -- unit-tested here."""

from __future__ import annotations

from pumpkinspice.relational import build_relational

ITEMS = [
    {
        "name": "Copper Dagger",
        "code": "copper_dagger",
        "level": 1,
        "type": "weapon",
        "subtype": "dagger",
        "craft": {
            "skill": "weaponcrafting",
            "level": 1,
            "items": [{"code": "copper", "quantity": 6}],
            "quantity": 1,
        },
    },
    {"name": "Copper", "code": "copper", "level": 1, "type": "resource"},  # no craft
]
MONSTERS = [{"code": "chicken", "drops": [{"code": "feather", "rate": 16}]}]
RESOURCES = [
    {
        "code": "copper_rocks",
        "skill": "mining",
        "level": 1,
        "drops": [{"code": "copper", "rate": 1}],
    }
]
MAPS = [
    {"x": 2, "y": 0, "content": {"type": "resource", "code": "copper_rocks"}},
    {"x": 0, "y": 0, "content": None},  # empty tile ignored
]


def test_build_relational_extracts_recipe_sources_locations() -> None:
    rows = build_relational(ITEMS, MONSTERS, RESOURCES, MAPS)

    assert {i["code"] for i in rows.items} == {"copper_dagger", "copper"}
    assert rows.item_craft == [
        {"item_code": "copper_dagger", "skill": "weaponcrafting", "craft_level": 1, "yield_qty": 1}
    ]
    assert rows.craft_ingredients == [
        {"item_code": "copper_dagger", "ingredient_code": "copper", "quantity": 6}
    ]
    # both monster and resource drops collapse into one sources table
    assert {
        "item_code": "feather",
        "source_type": "monster",
        "source_code": "chicken",
        "rate": 16,
    } in rows.sources
    assert {
        "item_code": "copper",
        "source_type": "resource",
        "source_code": "copper_rocks",
        "rate": 1,
    } in rows.sources
    # only tiles with content become locations
    assert rows.locations == [{"content_type": "resource", "code": "copper_rocks", "x": 2, "y": 0}]
