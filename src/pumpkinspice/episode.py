"""Episode-setup utilities shared by the CLI v2 runner (``v2run``) and the web trials runner.

Kept in CORE (not web) so both surfaces share ONE implementation of the two things a
per-episode/per-trial runner must get right: resetting a HeroBench character to a fresh
baseline (so episodes are IID) and building a genuinely-stochastic sampler (so episodes
actually diverge). Both were first written in web/runs.py; this is the shared home.
"""

from __future__ import annotations

import contextlib
from typing import Any

import httpx


def reset_herobench_character(base_url: str, character: str) -> None:
    """Reset a HeroBench character to a fresh L1 / empty-inventory baseline: delete then
    create (the REST delete clears the inventory hash too, verified 2026-06-29). The CREATE
    is verified (raise_for_status), so a caller cannot mistake a half-done reset -- character
    deleted but not recreated -- for success. The delete tolerates "did not exist" (498)."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=20.0) as c:
        with contextlib.suppress(httpx.HTTPError):  # delete: raw JSON-string body; 498 if absent
            c.post("/characters/delete", json=character)
        c.post("/characters/create", json={"name": character, "skin": "men2"}).raise_for_status()


def stochastic_sampler(temperature: float, seed: int) -> dict[str, Any]:
    """A genuinely stochastic, per-seed-reproducible sampler. The decoder's GREEDY default
    pins top_k=1 (which collapses to argmax, making temperature and seed INERT) and seed=0;
    override both so episodes actually diverge yet stay reproducible from (temperature, seed)."""
    return {"temperature": temperature, "top_k": 0, "top_p": 0.95, "seed": seed}
