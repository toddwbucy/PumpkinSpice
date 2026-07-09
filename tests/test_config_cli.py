"""Config loading, the logging helper, and the CLI entry points."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from pumpkinspice.cli import _load_env_local, main
from pumpkinspice.config import load_config
from pumpkinspice.logging import _coerce_level, configure_logging


def test_load_env_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env.local"
    env.write_text("# secrets\nPS_TEST_FOO=bar\nPS_TEST_KEEP=fromfile\n\n")
    monkeypatch.delenv("PS_TEST_FOO", raising=False)
    monkeypatch.setenv("PS_TEST_KEEP", "preset")
    _load_env_local(env)
    assert os.environ["PS_TEST_FOO"] == "bar"  # loaded from file
    assert os.environ["PS_TEST_KEEP"] == "preset"  # existing value is kept


OFFLINE = Path(__file__).resolve().parent.parent / "configs" / "offline.toml"


def test_load_config_and_accessors() -> None:
    cfg = load_config(OFFLINE)
    assert cfg.plugin_name("decoder") == "echo"
    assert cfg.slot_config("world")["character"] == "hero"
    assert cfg.max_turns == 6
    assert cfg.task


def test_plugin_name_missing_raises() -> None:
    cfg = load_config(OFFLINE)
    cfg.run.pop("decoder")
    with pytest.raises(KeyError):
        cfg.plugin_name("decoder")


def test_coerce_level() -> None:
    assert _coerce_level(logging.DEBUG) == logging.DEBUG
    assert _coerce_level("debug") == logging.DEBUG
    assert _coerce_level("nonsense") == logging.INFO


def test_cli_plugins_lists_all_slots(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["plugins"])
    out = capsys.readouterr().out
    assert rc == 0
    for slot in ("decoder", "retrieval", "world", "prompt", "capture"):
        assert slot in out


def test_cli_parity_needs_config_or_compare() -> None:
    # parity with neither --config nor --compare is a usage error (exit 2)
    assert main(["parity"]) == 2


def test_cli_transport_requires_config() -> None:
    # transport's --config is required; argparse exits with SystemExit(2)
    with pytest.raises(SystemExit):
        main(["transport"])


def test_cli_run_offline(tmp_path: Path) -> None:
    # Point the capture at tmp so the run does not write into the repo.
    cfg_text = OFFLINE.read_text().replace(
        'path = "captures/offline.jsonl"', f'path = "{tmp_path / "run.jsonl"}"'
    )
    cfg_file = tmp_path / "offline.toml"
    cfg_file.write_text(cfg_text)
    assert main(["run", "--config", str(cfg_file)]) == 0
    assert (tmp_path / "run.jsonl").exists()


def test_configure_logging_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("DEBUG")  # second call should not raise


def test_parse_model_spec() -> None:
    from pumpkinspice.cli import _parse_model_spec

    # ':N' = per-model max_tokens cap; omitted = 0 (unbounded); model ids keep '/'
    assert _parse_model_spec(
        "mistral-small-24b:256, qwen/qwen3.6-27b , gemma-4-26b-a4b-it:512"
    ) == [
        ("mistral-small-24b", 256),
        ("qwen/qwen3.6-27b", 0),
        ("gemma-4-26b-a4b-it", 512),
    ]


def test_cli_analyze(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    turn = {
        "index": 0,
        "world_state": {"level": 2, "x": 0, "y": 0, "xp": 5},
        "retrieval": {"backend": "pgvector"},
        "action": {"kind": "rest", "args": {}},
        "outcome": {"ok": True, "data": {"level": 2}},
        "timings_ms": {"decode": 10.0},
        "model": "m1",
    }
    cap = tmp_path / "r.jsonl"
    cap.write_text(json.dumps(turn))
    rc = main(["analyze", str(cap), "--goal-level", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "m1" in out and "steps" in out


def test_v2_episode_config(tmp_path: Path) -> None:
    pytest.importorskip("numpy")  # bench_herobench -> introspect/__init__ -> geometry (numpy)
    from pumpkinspice.cli import _v2_episode_config
    from pumpkinspice.introspect.bench_herobench import RAMP, V2_LADDER

    base = "configs/v2_smoke_chicken_qwen3_8b.toml"
    task = V2_LADDER["v2_yellow_slime"]
    cfg, path = _v2_episode_config(base, task, ep=3, seed=7, out_dir=tmp_path)
    assert cfg.run["task"] == task.task
    # ALL goal fields set from the task: a v2 tier -> monster goal, item/level cleared to None
    assert cfg.run["goal_monster"] == "yellow_slime"
    assert cfg.run["goal_item"] is None and cfg.run["goal_level"] is None
    # a genuinely-stochastic sampler for this seed (top_k un-pinned so the seed is not inert)
    s = cfg.slots["decoder"]["sampler"]
    assert s["seed"] == 7 and s["top_k"] == 0 and s["temperature"] == 0.7
    # filename keyed on the episode INDEX (+ seed), not the seed alone
    assert path == tmp_path / "v2_yellow_slime__ep003_seed007.jsonl"
    assert cfg.slots["capture"]["path"] == str(path)
    assert cfg.run["prompt"] == "react"  # base config's externalized react arm preserved

    # a RAMP tier sets its item goal and CLEARS the base config's monster goal (no stale leak)
    ramp = RAMP["copper_dagger"]  # goal_item, goal_monster is None
    cfg2, _ = _v2_episode_config(base, ramp, ep=0, seed=0, out_dir=tmp_path)
    assert cfg2.run["goal_item"] == "copper_dagger"
    assert cfg2.run["goal_monster"] is None  # base config's "chicken" overwritten -> no leak
