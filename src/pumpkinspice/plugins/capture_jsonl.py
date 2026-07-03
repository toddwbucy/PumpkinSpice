"""Per-turn capture to JSON Lines (spec section 7).

One JSON object per turn: rendered prompt, raw model output, retrieval calls +
latency, action, and outcome. Append-only and flushed per turn so a crashed run
still leaves a usable corpus. Shaped to align with the WeaverTools per-turn
record for weaver-analysis.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, TextIO

from ..contracts import Turn


class JsonlCapture:
    name = "jsonl"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        path = Path(config.get("path", "captures/run.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh: TextIO = path.open("a", encoding="utf-8")

    def record(self, turn: Turn) -> None:
        self._fh.write(json.dumps(dataclasses.asdict(turn), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
