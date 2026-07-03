"""pgvector retrieval: top-k semantic vector search, optionally enriched with
relational structure from the HeroBench schema.

Two ablation modes on one backend (select via the ``relational`` config flag):
  - semantic only (``relational = false``, default): a single cosine top-k.
  - semantic + relational (``relational = true``): after the cosine top-k, JOIN
    the kg.* relational tables to append the schema neighbours of each retrieved
    item -- its recipe ingredients, the monsters/resources that drop them, and
    where those are found. The capture's ``backend`` field records which mode ran
    ("pgvector" vs "pgvector+relational").

Both are conventional RAG: hand-written cosine and plain SQL joins over an
explicit schema. Neither delegates to HADES hybrid/rerank/structural retrieval.

Auth: connects with a SCOPED, READ-ONLY DSN read from an env var (``dsn_env``),
never the root ``POSTGRESQL_PASSWORD``. The read-only single-database role is the
enforcement that this backend cannot ingest, embed, or reach another database.

Requires the ``pgvector`` extra and a seeded table with an embedding column (and,
for relational mode, the kg.* tables from scripts/seed_relational_pg.py). Query
embedding comes from an OpenAI-compatible ``/v1/embeddings`` endpoint.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from ..contracts import BeliefNode, RetrievalResult

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier_parts(value: str, setting: str, *, qualified: bool = False) -> tuple[str, ...]:
    """Validate a config-supplied SQL identifier and return its parts.

    ``qualified=True`` allows one schema qualification ("kg.belief_nodes" ->
    ("kg", "belief_nodes")); every part must be a plain identifier. Belt and
    braces on top of psycopg's sql.Identifier quoting: identifiers come from
    operator config, but they still must never ride into SQL as raw text.
    """
    parts = tuple(value.split(".", 1)) if qualified else (value,)
    for part in parts:
        if not _IDENTIFIER_RE.match(part):
            raise ValueError(
                f"pgvector retrieval config {setting!r} is not a valid SQL identifier: {value!r}"
            )
    return parts


class PgVectorRetrieval:
    name = "pgvector"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        dsn_env = config.get("dsn_env", "PUMPKINSPICE_PG_DSN")
        dsn = os.environ.get(dsn_env)
        if not dsn:
            raise RuntimeError(
                f"pgvector retrieval needs a scoped read-only DSN in env ${dsn_env}. "
                f"Do NOT use the root POSTGRESQL_PASSWORD here."
            )
        self._dsn: str = dsn
        self.table = str(config.get("table", "belief_nodes"))
        self.id_column = str(config.get("id_column", "id"))
        self.text_column = str(config.get("text_column", "text"))
        self.metadata_column = str(config.get("metadata_column", "metadata"))
        self.vector_column = str(config.get("vector_column", "embedding"))
        self.relational = bool(config.get("relational", False))
        self.relational_schema = str(config.get("relational_schema", "kg"))
        # Validate every config-supplied identifier up front (a clear ValueError
        # at construction beats a malformed query at retrieve time). Only
        # ``table`` may be schema-qualified ("kg.belief_nodes").
        self._table_parts = _identifier_parts(self.table, "table", qualified=True)
        for setting, value in (
            ("id_column", self.id_column),
            ("text_column", self.text_column),
            ("metadata_column", self.metadata_column),
            ("vector_column", self.vector_column),
            ("relational_schema", self.relational_schema),
        ):
            _identifier_parts(value, setting)
        # How many of the top item hits get their full recipe-book entry appended.
        self.recipe_top_n = int(config.get("recipe_top_n", 3))
        self.embed_url = str(config.get("embed_url", "http://192.168.0.203:1234")).rstrip("/")
        self.embed_model = config.get("embed_model")
        self._embed_client = httpx.Client(base_url=self.embed_url, timeout=60.0)

        # Lazy import so the offline core installs without psycopg.
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from psycopg import sql as psycopg_sql
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "pgvector retrieval requires the 'pgvector' extra: uv sync --extra pgvector"
            ) from exc
        self._psycopg = psycopg
        self._register_vector = register_vector
        self._sql = psycopg_sql

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
        # Pass the query vector as a text literal cast to vector. A bare Python
        # list is sent as double precision[], for which no `<=>` operator exists.
        vec_literal = "[" + ",".join(repr(float(x)) for x in vec) + "]"
        # Cosine distance operator <=> ; score = 1 - distance. Identifiers are
        # composed with psycopg's sql module (quoted, never interpolated as
        # text); values stay parameterized.
        stmt = self._sql.SQL(
            "SELECT {id}, {text}, {metadata}, 1 - ({vector} <=> %s::vector) AS score "
            "FROM {table} ORDER BY {vector} <=> %s::vector LIMIT %s"
        ).format(
            id=self._sql.Identifier(self.id_column),
            text=self._sql.Identifier(self.text_column),
            metadata=self._sql.Identifier(self.metadata_column),
            vector=self._sql.Identifier(self.vector_column),
            table=self._sql.Identifier(*self._table_parts),
        )
        nodes: list[BeliefNode] = []
        with self._psycopg.connect(self._dsn) as conn:
            self._register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(stmt, (vec_literal, vec_literal, top_k))
                for row in cur.fetchall():
                    meta = row[2] if isinstance(row[2], dict) else {}
                    nodes.append(
                        BeliefNode(id=str(row[0]), text=row[1], score=float(row[3]), metadata=meta)
                    )
                if self.relational:
                    item_codes = [
                        m["code"]
                        for n in nodes
                        if (m := n.metadata).get("kind") == "item" and m.get("code")
                    ]
                    # Cap to the top item hits: the goal item's recipe is what matters;
                    # appending every hit's full recipe bloats the prompt and adds noise.
                    nodes.extend(self._relational_facts(cur, item_codes[: self.recipe_top_n]))

        backend = self.name + ("+relational" if self.relational else "")
        return RetrievalResult(
            query=query,
            nodes=nodes,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            backend=backend,
        )

    def _relational_facts(self, cur: Any, item_codes: list[str]) -> list[BeliefNode]:
        """Append the precomputed RECIPE BOOK entry for each semantic-hit item: the
        fully-flattened crafting chain -- skills + required levels, rolled-up gather
        quantities, and gather/craft locations (built by scripts/seed_recipe_book.py).

        This is a plain lookup; the multi-hop graph walk that resolves a *crafted*
        ingredient down to its ore + location happened build-side. A naive one-hop
        join here would dead-end at the crafted ingredient (e.g. `copper`) and never
        reach the ore or its coordinates -- which sandbagged the control. Still
        conventional RAG (a lookup over our own table), NOT HADES structural retrieval.
        """
        if not item_codes:
            return []
        # recipe_book lives in the (validated) relational schema; compose the
        # qualified identifier rather than interpolating the schema as text.
        stmt = self._sql.SQL(
            "SELECT item_code, recipe_text FROM {table} WHERE item_code = ANY(%s)"
        ).format(table=self._sql.Identifier(self.relational_schema, "recipe_book"))
        cur.execute(stmt, (item_codes,))
        return [
            BeliefNode(
                id=f"recipe_book:{item}",
                text=text,
                metadata={"relation": "recipe_book", "item": item},
            )
            for item, text in cur.fetchall()
        ]
