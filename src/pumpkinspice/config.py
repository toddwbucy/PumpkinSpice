"""Run configuration: which plugin fills each slot, plus per-plugin settings.

A run config is TOML. The top-level ``[run]`` table selects the plugin name for
each slot; a ``[<slot>]`` table holds that plugin's settings. Example:

    [run]
    decoder   = "lmstudio"
    retrieval = "pgvector"
    world     = "herobench"
    prompt    = "default"
    capture   = "jsonl"
    task      = "Reach level 5 by fighting chickens."
    max_turns = 20

    [decoder]
    base_url = "http://192.168.0.203:1234"
    model    = "qwen3-8b"

    [retrieval]
    top_k = 5
    dsn_env = "PUMPKINSPICE_PG_DSN"   # scoped, read-only -- never the root creds
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .kernel import SLOTS


@dataclass
class RunConfig:
    run: dict[str, Any] = field(default_factory=dict)
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def plugin_name(self, slot: str) -> str:
        try:
            return str(self.run[slot])
        except KeyError:
            raise KeyError(f"[run] is missing a plugin selection for slot {slot!r}") from None

    def slot_config(self, slot: str) -> dict[str, Any]:
        return self.slots.get(slot, {})

    @property
    def task(self) -> str:
        return str(self.run.get("task", ""))

    @property
    def max_turns(self) -> int:
        return int(self.run.get("max_turns", 10))


def load_config(path: str | Path) -> RunConfig:
    data = tomllib.loads(Path(path).read_text())
    run = dict(data.get("run", {}))
    slots = {slot: dict(data.get(slot, {})) for slot in SLOTS}
    return RunConfig(run=run, slots=slots)
