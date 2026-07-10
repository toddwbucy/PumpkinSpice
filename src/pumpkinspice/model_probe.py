"""Shared model-metadata probe for capture provenance.

Both the decoder plugins (Turn.model_info) and the decoder-parity gate
(lmstudio_model_info) need to fetch a served model's environment -- precision,
architecture, the context length it actually loaded at -- from an
OpenAI-compatible ``/models`` endpoint, matched by id. This is that one query, so
the two provenance sites cannot drift (and both get the same `state`-aware
selection and short timeout).

Best-effort by construction: a short, dedicated timeout (provenance must NEVER
block a run -- a decoder client's own timeout is the full generation budget, up to
1800s in the MATH config, and a hung models endpoint would otherwise stall the run
at the gate) and ``None`` on any error.
"""

from __future__ import annotations

from typing import Any

import httpx

# Provenance is nice-to-have; a slow/hung models endpoint must not stall a run behind
# the full generation timeout. Query with a short, dedicated ceiling.
PROBE_TIMEOUT_S = 5.0


def fetch_model_entry(
    client: httpx.Client,
    endpoint: str,
    model: str | None,
    *,
    timeout: float = PROBE_TIMEOUT_S,
) -> dict[str, Any] | None:
    """GET an OpenAI-compatible models ``endpoint`` and return the raw entry for ``model``
    (matched by id). When ``model`` is None (LMStudio's "decode whatever is loaded"
    default), prefer the entry whose ``state`` marks it loaded, else fall back to the
    first -- so a server that lists every downloaded model does not describe an unloaded,
    different one. Best-effort: returns None on any error, with a short timeout so it never
    blocks the run.
    """
    try:
        resp = client.get(endpoint, timeout=timeout)
        resp.raise_for_status()
        data: list[dict[str, Any]] = resp.json().get("data", [])
    except Exception:  # provenance discovery must never break a run
        return None
    if model:
        return next((m for m in data if m.get("id") == model), None)
    # No model configured: prefer an entry explicitly marked loaded (LMStudio `state`),
    # else the first. vLLM's /v1/models has no `state` and lists only the served model,
    # so the first-entry fallback is correct there.
    loaded = next((m for m in data if m.get("state") == "loaded"), None)
    if loaded is not None:
        return loaded
    return data[0] if data else None
