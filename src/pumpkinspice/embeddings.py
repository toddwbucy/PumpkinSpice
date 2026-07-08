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
from typing import Any

# Headless embeddings default. MUST match the model that seeded the corpus -- re-run
# seed_corpus.py AND seed_corpus_arango.py if you change it (query and document vectors
# must share one space, per arm).
DEFAULT_EMBED_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Node-metadata key the seeders stamp with the embed model that produced the vectors,
# so a retrieval arm can fail fast on a query/document embedder mismatch.
EMBED_MODEL_META_KEY = "embed_model"


def assert_embed_model_matches(corpus_model: str | None, configured_model: str | None) -> None:
    """Fail fast if the corpus was seeded with a different embed model than retrieval is
    configured to use. A 768-dim mismatch raises no error in the cosine search -- it
    silently returns worse neighbours, misattributed to the model under test. A corpus
    with no stamp (pre-provenance) or an unreadable one passes ``corpus_model=None`` and
    skips the check."""
    if corpus_model and corpus_model != configured_model:
        raise ValueError(
            f"embed-model mismatch: corpus was seeded with {corpus_model!r} but retrieval "
            f"is configured with {configured_model!r} -- query and document vectors must "
            "share one embedder. Re-seed (seed_corpus*.py) or fix the config."
        )


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
