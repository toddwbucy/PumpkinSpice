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

# Exact cosine top-k. Distances computed against the stored embedding array.
_AQL = """
FOR doc IN @@collection
  LET dot = SUM(FOR i IN 0..(LENGTH(@q) - 1) RETURN doc.embedding[i] * @q[i])
  LET nd = SQRT(SUM(FOR x IN doc.embedding RETURN x * x))
  LET score = nd == 0 ? 0 : dot / (@nq * nd)
  SORT score DESC
  LIMIT @k
  RETURN {id: doc._key, text: doc.text, score: score}
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

        self.embed_url = str(config.get("embed_url", "http://192.168.0.203:1234")).rstrip("/")
        self.embed_model = config.get("embed_model")
        self._embed_client = httpx.Client(base_url=self.embed_url, timeout=60.0)

        try:
            from arango import ArangoClient
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "arango retrieval requires the 'arango' extra: uv sync --extra arango"
            ) from exc
        self._db = ArangoClient(hosts=self.url).db(
            self.database, username=self._user, password=self._password
        )

    def _embed(self, query: str) -> list[float]:
        payload: dict[str, Any] = {"input": query}
        if self.embed_model:
            payload["model"] = self.embed_model
        resp = self._embed_client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        return [float(x) for x in resp.json()["data"][0]["embedding"]]

    def retrieve(self, query: str, *, top_k: int) -> RetrievalResult:
        t0 = time.perf_counter()
        vec = self._embed(query)
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
