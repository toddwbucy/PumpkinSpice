"""Belief-node rendering is pure and deterministic, so it is unit-tested here."""

from __future__ import annotations

from pathlib import Path

from pumpkinspice.corpus import (
    load_corpus,
    render_item,
    render_locations,
    render_mechanics,
    render_monster,
    render_resource,
)


def test_render_item_with_craft() -> None:
    node = render_item(
        {
            "name": "Cooked Gudgeon",
            "code": "cooked_gudgeon",
            "level": 1,
            "type": "consumable",
            "subtype": "restore",
            "description": "Restores 8HP.",
            "effects": [{"name": "restore", "value": 8}],
            "craft": {
                "skill": "cooking",
                "level": 1,
                "items": [{"code": "gudgeon", "quantity": 1}],
                "quantity": 1,
            },
        }
    )
    assert node.id == "item:cooked_gudgeon"
    assert "Cooked Gudgeon" in node.text
    assert "restore 8" in node.text
    assert "1x gudgeon" in node.text
    assert node.metadata == {"kind": "item", "code": "cooked_gudgeon", "level": 1}


def test_render_monster_filters_zero_stats() -> None:
    node = render_monster(
        {
            "name": "Chicken",
            "code": "chicken",
            "level": 1,
            "hp": 60,
            "attack_fire": 0,
            "attack_water": 4,
            "attack_air": 0,
            "res_fire": 0,
            "res_water": 0,
            "min_gold": 0,
            "max_gold": 1,
            "drops": [{"code": "egg", "rate": 25, "min_quantity": 1, "max_quantity": 1}],
        }
    )
    assert node.id == "monster:chicken"
    assert "water 4" in node.text  # nonzero attack kept
    assert "fire" not in node.text  # zero stats dropped
    assert "egg (1/25)" in node.text
    assert "Gold 0-1" in node.text


def test_render_resource() -> None:
    node = render_resource(
        {
            "name": "Ash Tree",
            "code": "ash_tree",
            "skill": "woodcutting",
            "level": 1,
            "drops": [{"code": "ash_wood", "rate": 1}, {"code": "sap", "rate": 5000}],
        }
    )
    assert node.id == "resource:ash_tree"
    assert "woodcutting level 1" in node.text
    assert "ash_wood, sap" in node.text


def test_render_locations_aggregates_by_content() -> None:
    maps = [
        {"x": -5, "y": -5, "content": {"type": "monster", "code": "ogre"}},
        {"x": 2, "y": 3, "content": {"type": "monster", "code": "ogre"}},
        {"x": 0, "y": 0, "content": None},  # empty tile skipped
        {"x": 1, "y": 1},  # no content key skipped
    ]
    nodes = render_locations(maps)
    assert len(nodes) == 1
    node = nodes[0]
    assert node.id == "location:monster:ogre"
    assert node.metadata["count"] == 2
    assert "(-5,-5)" in node.text and "(2,3)" in node.text


def test_render_mechanics_teaches_leveling_via_crafting() -> None:
    """The corpus documents craft REQUIREMENTS but nothing crafts a skill's XP --
    that mechanic is only discoverable from a live server response. These nodes
    are the fix: a worked example (copper_dagger, the cheapest weaponcrafting
    recipe) generalized to any skill/high-level-recipe pair like sticky_sword."""
    nodes = render_mechanics()
    ids = {n.id for n in nodes}
    assert {
        "mechanic:crafting_xp",
        "mechanic:leveling_via_crafting",
        "mechanic:combat_risk",
    } <= ids
    leveling = next(n for n in nodes if n.id == "mechanic:leveling_via_crafting")
    assert "copper_dagger" in leveling.text and "sticky_sword" in leveling.text
    assert "weaponcrafting level 5" in leveling.text
    for n in nodes:
        assert n.metadata["kind"] == "mechanic"


def test_load_corpus_includes_mechanics(tmp_path: Path) -> None:
    for name in ("items.json", "monsters.json", "resources.json", "maps.json"):
        (tmp_path / name).write_text("[]")
    nodes = load_corpus(tmp_path)
    assert {n.id for n in nodes} == {n.id for n in render_mechanics()}
