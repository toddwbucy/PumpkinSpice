"""ArangoDB retrieval: plain top-k cosine search over a document collection.

The second ablation backend (a KG-style store with semantic structure). This is
conventional RAG: an exact brute-force cosine top-k computed in AQL, hand-written
-- it MUST NOT use the HADES hybrid/rerank/structural retrieval, and it uses this
repo's own direct HTTP client, NOT the HADES roproxy/rwproxy unix-socket path.

Exact cosine over the corpus is fine at this scale (~300 docs); ArangoDB 3.12's
native vector index is experimental and not relied on here.

Auth: connects with a SCOPED, READ-ONLY user from env (``user_env`` /
``password_env``), never the Arango root password. The read-only grant (and the
user's default "no access" to every other database) is the enforcement.

Requires the ``arango`` extra (``uv sync --extra arango``) and a seeded
collection. Query embeddings come from an OpenAI-compatible ``/v1/embeddings``
endpoint (e.g. LMStudio).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from ..contracts import BeliefNode, RetrievalResult
from ..embeddings import (
    CHECK_MODES,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    EMBED_MODEL_META_KEY,
    check_embed_provenance,
    embed_query,
    warm_up,
)

# Exact cosine top-k. Distances computed against the stored embedding array.
_AQL = """
FOR doc IN @@collection
  LET dot = SUM(FOR i IN 0..(LENGTH(@q) - 1) RETURN doc.embedding[i] * @q[i])
  LET nd = SQRT(SUM(FOR x IN doc.embedding RETURN x * x))
  LET score = nd == 0 ? 0 : dot / (@nq * nd)
  SORT score DESC
  LIMIT @k
  RETURN {id: doc.id != null ? doc.id : doc._key, text: doc.text, score: score}
"""


class ArangoRetrieval:
    name = "arango"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.url = str(config.get("url", "http://localhost:8529"))
        self.database = config.get("database", "herobench_kg")
        self.collection = config.get("collection", "belief_nodes")
        user_env = config.get("user_env", "ARANGO_AGENT_USER")
        password_env = config.get("password_env", "ARANGO_AGENT_PASSWORD")
        self._user = os.environ.get(user_env)
        self._password = os.environ.get(password_env)
        if not self._user or not self._password:
            raise RuntimeError(
                f"arango retrieval needs a scoped read-only user in env "
                f"${user_env}/${password_env}. Do NOT use the Arango root password."
            )
        if self._user == "root":
            raise RuntimeError(
                "arango retrieval must not run as root; use the scoped read-only user."
            )

        self.embed_url = str(config.get("embed_url", DEFAULT_EMBED_URL)).rstrip("/")
        self.embed_model = config.get("embed_model", DEFAULT_EMBED_MODEL)
        self._check_mode = str(config.get("embed_model_check", "strict"))
        if self._check_mode not in CHECK_MODES:
            raise ValueError(
                f"embed_model_check must be one of {CHECK_MODES}; got {self._check_mode!r}"
            )
        self._provenance_checked = False
        self._embed_client = httpx.Client(base_url=self.embed_url, timeout=60.0)
        warm_up(self._embed_client, self.embed_model)  # avoid cold-start in latency

        try:
            from arango import ArangoClient
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "arango retrieval requires the 'arango' extra: uv sync --extra arango"
            ) from exc
        self._db = ArangoClient(hosts=self.url).db(
            self.database, username=self._user, password=self._password
        )
        # The embed-provenance check runs lazily on the first retrieve(), not here.

    def _fetch_provenance(self) -> tuple[list[str], int | None]:
        """(distinct embed-model stamps across the corpus, one stored vector's dim).
        DISTINCT (not LIMIT 1) so a partial re-seed's mixed stamps are caught; the key is
        BOUND via @key so it tracks EMBED_MODEL_META_KEY instead of a hardcoded path. The
        dimension guards the arango arm specifically -- its AQL cosine has none, so a
        cross-dim query would truncate the dot product and degrade silently."""
        stamp_cursor: Any = self._db.aql.execute(
            "FOR doc IN @@collection RETURN DISTINCT doc.metadata[@key]",
            bind_vars={"@collection": self.collection, "key": EMBED_MODEL_META_KEY},
        )
        stamps = [s for s in stamp_cursor if isinstance(s, str)]
        dim_cursor: Any = self._db.aql.execute(
            "FOR doc IN @@collection LIMIT 1 RETURN LENGTH(doc.embedding)",
            bind_vars={"@collection": self.collection},
        )
        dim = next(iter(dim_cursor), None)
        return stamps, (int(dim) if isinstance(dim, int) else None)

    def _embed(self, query: str) -> list[float]:
        return embed_query(self._embed_client, self.embed_model, query)

    def retrieve(self, query: str, *, top_k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        vec = self._embed(query)
        if not self._provenance_checked:
            check_embed_provenance(
                self._fetch_provenance,
                self.embed_model,
                len(vec),
                mode=self._check_mode,
                backend="arango",
            )
            self._provenance_checked = True
        nq = math.sqrt(sum(x * x for x in vec)) or 1.0
        bind_vars: dict[str, Any] = {
            "@collection": self.collection,
            "q": vec,
            "nq": nq,
            "k": top_k,
        }
        # execute() is typed as Cursor | Job | None; in plain mode it is a Cursor.
        cursor: Any = self._db.aql.execute(_AQL, bind_vars=bind_vars)
        nodes = [
            BeliefNode(id=str(row["id"]), text=row["text"], score=float(row["score"]))
            for row in cursor
        ]
        return RetrievalResult(
            query=query,
            nodes=nodes,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            backend=self.name,
        )
