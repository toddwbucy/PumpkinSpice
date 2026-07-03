"""App-wide settings: the model under test + decode defaults (web PRD Phase 2).

Persisted to ``captures/settings.json`` and used as defaults by both Chat and
benchmark runs, so "the model under test" and how it is driven (temperature,
max_tokens cap, history window) are configured in one place. Empty/zero values
mean "do not override" -- model "" uses whatever LMStudio has loaded; max_tokens 0
is unbounded; history_window 0 is full history.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class AppSettings:
    model: str = ""  # "" -> use whatever LMStudio has loaded
    temperature: float = 0.0
    max_tokens: int = 0  # 0 -> unbounded
    history_window: int = 0  # 0 -> full history


_FIELDS = {f.name for f in dataclasses.fields(AppSettings)}


def _clamp(s: AppSettings) -> AppSettings:
    """Keep values in their valid ranges so a bad input can never reach LMStudio
    (a negative temperature is a 400 -- "must be >= 0"). temperature is the
    OpenAI-compatible [0, 2]; the counts are never negative."""
    s.temperature = min(2.0, max(0.0, s.temperature))
    s.max_tokens = max(0, s.max_tokens)
    s.history_window = max(0, s.history_window)
    return s


class SettingsStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> AppSettings:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                return _clamp(AppSettings(**{k: v for k, v in data.items() if k in _FIELDS}))
            except (ValueError, TypeError):
                pass
        return AppSettings()

    def update(self, **changes: object) -> AppSettings:
        """Partial update: only known, non-None fields are applied, then saved."""
        cur = self.get()
        for k, v in changes.items():
            if v is not None and k in _FIELDS:
                setattr(cur, k, v)
        cur = _clamp(cur)
        self.path.write_text(json.dumps(dataclasses.asdict(cur), indent=2))
        return cur
