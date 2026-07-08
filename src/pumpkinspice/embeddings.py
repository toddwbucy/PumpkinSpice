"""Shared embeddings config + client for the retrieval arms and the seeders.

Both retrieval plugins (pgvector, arango) and both seeders (seed_corpus{,_arango}.py)
embed text via an OpenAI-compatible ``/v1/embeddings`` endpoint. The default is headless
Ollama serving ``nomic-embed-text`` (see CLAUDE.md). Keeping the URL/model in ONE place
is the point: a mismatch between the two ablation arms -- or between query and document
vectors -- raises no error (all nomic variants are 768-dim), it silently degrades
retrieval, which would then be misattributed to the model under test. So the default
lives here once, not stamped literally across five call sites.

The URL/model are a UNIT: change one and you must change the other (and re-seed the
corpus, since documents were embedded with a specific model). Override both via config
(retrieval plugins), CLI flags (seeders), or ``PUMPKINSPICE_EMBED_URL`` /
``PUMPKINSPICE_EMBED_MODEL`` (web backend).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Headless embeddings default. MUST match the model that seeded the corpus -- re-run
# seed_corpus.py AND seed_corpus_arango.py if you change it (query and document vectors
# must share one space, per arm).
DEFAULT_EMBED_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Node-metadata key the seeders stamp with the embed model that produced the vectors,
# so a retrieval arm can catch a query/document embedder mismatch.
EMBED_MODEL_META_KEY = "embed_model"

# Provenance-check policy: raise on mismatch, only warn, or skip entirely. "off"/"warn"
# are the escape hatch for a known-same-space rename (an embedder whose name differs
# across servers, e.g. Ollama "nomic-embed-text" vs LMStudio "text-embedding-...").
CHECK_MODES = ("strict", "warn", "off")

# (distinct non-null embed-model stamps in the corpus, one stored embedding's dimension).
ProvenanceFetch = Callable[[], "tuple[list[str], int | None]"]


def check_embed_provenance(
    fetch: ProvenanceFetch,
    configured_model: str | None,
    query_dim: int | None,
    *,
    mode: str = "strict",
    backend: str = "retrieval",
) -> None:
    """Verify the corpus was embedded into the SAME space this retrieval arm queries.

    Called lazily on the first ``retrieve`` (so construction does no DB IO), reusing that
    query's connection. ``fetch`` returns the DISTINCT embed-model stamps across the whole
    corpus and one stored vector's dimension; it may raise (a DB/config fault) -- that is
    LOGGED and the check skipped, never silently swallowed. Checks honour ``mode``
    ("strict" raises, "warn" logs, "off" skips):

    * dimension -- definitive and name-independent (the headline "768-dim" hazard; the
      arango arm computes cosine with no dimension guard, so a cross-dim query degrades
      silently there);
    * mixed corpus -- >1 distinct stamp means a partial/aborted re-seed left mixed-space
      vectors (a plain ``LIMIT 1`` probe would pass or fail on which row it happened to hit);
    * name mismatch -- the stamp vs the configured model.

    A falsy ``configured_model`` is the "server picks its loaded model" mode: unverifiable,
    so skip (symmetric with an unstamped corpus)."""
    if mode == "off" or not configured_model:
        return
    try:
        stamps, stored_dim = fetch()
    except Exception as exc:
        log.warning("%s: embed-provenance check SKIPPED (corpus read failed: %s)", backend, exc)
        return
    stamps = [s for s in stamps if s]  # drop null/empty stamps
    if query_dim and stored_dim and query_dim != stored_dim:
        _report(
            f"{backend}: embedding DIMENSION mismatch -- corpus is {stored_dim}-dim but the "
            f"query embedder is {query_dim}-dim. Re-seed, or point at the seeding embedder.",
            mode,
        )
        return
    if len(stamps) > 1:
        _report(
            f"{backend}: corpus has MIXED embed models {sorted(stamps)} -- a partial re-seed "
            "left vectors in different spaces. Re-seed the whole corpus.",
            mode,
        )
        return
    if stamps and stamps[0] != configured_model:
        _report(
            f"{backend}: embed-model mismatch -- corpus seeded with {stamps[0]!r} but retrieval "
            f"configured with {configured_model!r}. Re-seed, fix the config, or set "
            "embed_model_check='off'/'warn' if you know they share one space.",
            mode,
        )


def _report(msg: str, mode: str) -> None:
    """Raise (strict) or log (warn) a failed provenance check."""
    if mode == "warn":
        log.warning(msg)
    else:
        raise ValueError(msg)


def embed_query(client: Any, model: str | None, query: str) -> list[float]:
    """Embed one query via an OpenAI-compatible ``/v1/embeddings`` endpoint.

    ``client`` is an httpx.Client with base_url set to the embed endpoint. ``model``
    is sent when set (Ollama requires it; an LMStudio-style server may pick its loaded
    model when omitted)."""
    payload: dict[str, Any] = {"input": query}
    if model:
        payload["model"] = model
    resp = client.post("/v1/embeddings", json=payload)
    resp.raise_for_status()
    return [float(x) for x in resp.json()["data"][0]["embedding"]]


def warm_up(client: Any, model: str | None) -> None:
    """Best-effort warm-up embed at construction so a cold model load is not billed
    into the first ``retrieve()``'s latency. Ollama unloads a model after its
    keep-alive (~5 min default; set ``OLLAMA_KEEP_ALIVE=-1`` for scored latency runs),
    and reasoning-model decode gaps easily exceed that. Never fatal -- a down embedder
    surfaces on the first real retrieve()."""
    with contextlib.suppress(Exception):
        embed_query(client, model, "warm up")
