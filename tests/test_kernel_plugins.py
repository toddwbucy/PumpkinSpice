"""Smoke tests: plugin discovery and an end-to-end offline run.

These need no external services -- they use the echo decoder, null retrieval,
and mock world, so CI can run them anywhere.
"""

from __future__ import annotations

from pathlib import Path

from pumpkinspice import kernel
from pumpkinspice.config import load_config
from pumpkinspice.loop import AgentLoop, parse_action

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "offline.toml"


def test_all_slots_have_plugins():
    found = kernel.discover()
    assert set(found) == set(kernel.SLOTS)
    # Built-ins registered via entry points AND shipped in this tree. Backends
    # that are registered but land in later PRs (pgvector/arango retrieval,
    # plan/replan/executor prompts, hanoi world) are asserted by their own
    # test modules when their code arrives.
    assert "echo" in found["decoder"] and "lmstudio" in found["decoder"]
    assert "null" in found["retrieval"]
    assert "mock" in found["world"] and "herobench" in found["world"]


def test_parse_action_extracts_json():
    a = parse_action('blah\n{"action": "move", "args": {"x": 2, "y": 3}}\ntrailing')
    assert a.kind == "move" and a.args == {"x": 2, "y": 3}


def test_parse_action_falls_back_to_rest():
    assert parse_action("no json here").kind == "rest"


def test_offline_run_produces_captures(tmp_path):
    cfg = load_config(CONFIG)
    # redirect capture into tmp so the test does not write the repo
    cfg.slots["capture"]["path"] = str(tmp_path / "run.jsonl")
    parts = {
        slot: kernel.load_plugin(slot, cfg.plugin_name(slot), cfg.slot_config(slot))
        for slot in kernel.SLOTS
    }
    loop = AgentLoop(
        decoder=parts["decoder"],
        retrieval=parts["retrieval"],
        world=parts["world"],
        prompt=parts["prompt"],
        capture=parts["capture"],
        task=cfg.task,
        top_k=5,
    )
    turns = loop.play(cfg.max_turns)
    assert len(turns) == cfg.max_turns
    # the scripted move then fights should have advanced the mock world
    assert turns[0].action["kind"] == "move"
    assert (tmp_path / "run.jsonl").read_text().count("\n") == cfg.max_turns


def test_play_honors_should_stop(tmp_path):
    cfg = load_config(CONFIG)
    cfg.slots["capture"]["path"] = str(tmp_path / "run.jsonl")
    parts = {
        slot: kernel.load_plugin(slot, cfg.plugin_name(slot), cfg.slot_config(slot))
        for slot in kernel.SLOTS
    }
    loop = AgentLoop(
        decoder=parts["decoder"],
        retrieval=parts["retrieval"],
        world=parts["world"],
        prompt=parts["prompt"],
        capture=parts["capture"],
        task=cfg.task,
        top_k=5,
    )
    seen = {"n": 0}

    def should_stop() -> bool:
        seen["n"] += 1
        return seen["n"] > 2  # let two turns run, then stop before the third

    turns = loop.play(10, should_stop=should_stop)  # max_turns 10 NOT reached
    assert len(turns) == 2
