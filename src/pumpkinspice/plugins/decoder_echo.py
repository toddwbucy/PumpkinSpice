"""Offline decoder stub: lets the loop run end to end with no LMStudio.

It emits a deterministic, schema-valid action so the harness is exercisable
without any external service. Configure ``script`` to drive specific actions;
otherwise it rests.
"""

from __future__ import annotations

import json
from typing import Any


class EchoDecoder:
    name = "echo"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # Optional list of {"action":..., "args":...} dicts, replayed in order.
        self._script: list[dict[str, Any]] = list(config.get("script", []))
        self._i = 0

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        if self._i < len(self._script):
            step = self._script[self._i]
            self._i += 1
        else:
            step = {"action": "rest", "args": {}}
        return "Thinking: offline echo decoder.\n" + json.dumps(step)
