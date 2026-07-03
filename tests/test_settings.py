"""App settings store (model under test + decode defaults)."""

from __future__ import annotations

from pathlib import Path

from pumpkinspice.web.settings import AppSettings, SettingsStore


def test_defaults_when_absent(tmp_path: Path) -> None:
    s = SettingsStore(tmp_path / "settings.json").get()
    assert s == AppSettings(model="", temperature=0.0, max_tokens=0, history_window=0)


def test_partial_update_persists(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.update(model="mistral-24b", max_tokens=256)
    # only the given fields change; the rest keep defaults
    assert store.get() == AppSettings(model="mistral-24b", max_tokens=256)
    # a second partial update keeps the earlier ones (None is ignored)
    store.update(temperature=0.7, model=None)  # type: ignore[arg-type]
    s = store.get()
    assert s.model == "mistral-24b" and s.temperature == 0.7 and s.max_tokens == 256
    # survives a fresh store (persisted to disk)
    assert SettingsStore(tmp_path / "settings.json").get().model == "mistral-24b"


def test_unknown_keys_ignored(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text('{"model": "m", "bogus": 1}')
    assert SettingsStore(p).get().model == "m"  # bogus ignored, no crash


def test_clamps_out_of_range(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    # a negative temperature is a hard LMStudio 400; clamp to 0 on save
    s = store.update(temperature=-0.7, max_tokens=-5, history_window=-1)
    assert s.temperature == 0.0 and s.max_tokens == 0 and s.history_window == 0
    assert store.update(temperature=5.0).temperature == 2.0  # capped at the [0, 2] max
    # and a bad value already on disk is corrected on read, not propagated
    p = tmp_path / "bad.json"
    p.write_text('{"temperature": -3}')
    assert SettingsStore(p).get().temperature == 0.0
