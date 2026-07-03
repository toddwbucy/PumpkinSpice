"""The microkernel: discover and instantiate plugins by entry-point group+name.

The kernel knows the five slots and nothing about any concrete backend. It
resolves a plugin by ``(group, name)`` through ``importlib.metadata`` and
constructs it with its config subsection. If a named plugin's package is not
installed, loading fails loudly here -- which is exactly how the build-side /
runtime airgap is enforced for a backend like HADES.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any

# The plugin slots and their entry-point groups. Adding a slot is a one-line
# change here plus a contract Protocol.
SLOTS: dict[str, str] = {
    "decoder": "pumpkinspice.decoder",
    "retrieval": "pumpkinspice.retrieval",
    "world": "pumpkinspice.world",
    "prompt": "pumpkinspice.prompt",
    "capture": "pumpkinspice.capture",
}


def available(slot: str) -> dict[str, EntryPoint]:
    """All registered plugins for a slot, keyed by name."""
    group = SLOTS[slot]
    return {ep.name: ep for ep in entry_points(group=group)}


def discover() -> dict[str, list[str]]:
    """Map every slot to the plugin names currently installed/registered."""
    return {slot: sorted(available(slot)) for slot in SLOTS}


class PluginError(RuntimeError):
    pass


def load_plugin(slot: str, name: str, config: dict[str, Any] | None = None) -> Any:
    """Resolve and construct one plugin. ``config`` is passed to its constructor."""
    if slot not in SLOTS:
        raise PluginError(f"unknown plugin slot {slot!r}; known: {sorted(SLOTS)}")
    eps = available(slot)
    ep = eps.get(name)
    if ep is None:
        raise PluginError(
            f"no {slot} plugin named {name!r}. Installed {slot} plugins: "
            f"{sorted(eps) or '(none)'}. If you expected one, its package may "
            f"not be installed (e.g. an extra: `uv sync --extra <backend>`)."
        )
    try:
        cls = ep.load()
    except Exception as exc:  # missing optional dep, import error, etc.
        raise PluginError(f"failed to import {slot} plugin {name!r} ({ep.value}): {exc}") from exc
    return cls(config or {})
