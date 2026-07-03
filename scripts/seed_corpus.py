#!/usr/bin/env python
"""Build-side seeder: embed the HeroBench encyclopedia into kg.belief_nodes.

Reads HeroBench's Data/*.json, renders belief nodes (pumpkinspice.corpus),
embeds each via an OpenAI-compatible /v1/embeddings endpoint, and upserts them
into kg.belief_nodes using the READ-WRITE loader role.

Auth: connects with the scoped loader DSN from $PUMPKINSPICE_PG_LOADER_DSN
(rw, build-side) -- NOT the agent's read-only DSN and NOT the root password.
Idempotent: re-running updates rows by id (ON CONFLICT).

Run:
  uv run --extra pgvector python scripts/seed_corpus.py \
      --data-dir ~/git/HeroBench/Virtual_Environment/Data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

# Make the package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pumpkinspice.corpus import CorpusNode, load_corpus

DEFAULT_DATA_DIR = Path.home() / "git/HeroBench/Virtual_Environment/Data"
UPSERT = """
INSERT INTO kg.belief_nodes (id, text, metadata, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
   SET text = EXCLUDED.text,
       metadata = EXCLUDED.metadata,
       embedding = EXCLUDED.embedding
"""


def embed_batch(client: httpx.Client, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.post("/v1/embeddings", json={"model": model, "input": texts})
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [[float(x) for x in d["embedding"]] for d in data]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--dsn-env", default="PUMPKINSPICE_PG_LOADER_DSN")
    ap.add_argument("--embed-url", default="http://192.168.0.203:1234")
    ap.add_argument("--embed-model", default="text-embedding-nomic-embed-text-v1.5")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        print(
            f"error: ${args.dsn_env} not set (run scripts/bootstrap_pg.py first).", file=sys.stderr
        )
        return 1
    if not args.data_dir.exists():
        print(f"error: data dir not found: {args.data_dir}", file=sys.stderr)
        return 1

    nodes: list[CorpusNode] = load_corpus(args.data_dir)
    print(f"rendered {len(nodes)} belief nodes from {args.data_dir}")

    embed_client = httpx.Client(base_url=args.embed_url.rstrip("/"), timeout=120.0)
    rows: list[tuple[str, str, Jsonb, list[float]]] = []
    for start in range(0, len(nodes), args.batch_size):
        batch = nodes[start : start + args.batch_size]
        vectors = embed_batch(embed_client, args.embed_model, [n.text for n in batch])
        if len(vectors) != len(batch):
            print(
                f"error: embedder returned {len(vectors)} vectors for {len(batch)} inputs",
                file=sys.stderr,
            )
            return 1
        for node, vec in zip(batch, vectors, strict=True):
            rows.append((node.id, node.text, Jsonb(node.metadata), vec))
        print(f"  embedded {min(start + args.batch_size, len(nodes))}/{len(nodes)}")

    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(UPSERT, rows)
            cur.execute("SELECT count(*) FROM kg.belief_nodes")
            total = cur.fetchone()[0]
    print(f"upserted {len(rows)} nodes; kg.belief_nodes now has {total} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
