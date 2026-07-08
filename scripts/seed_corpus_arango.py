#!/usr/bin/env python
"""Build-side seeder: embed the HeroBench encyclopedia into ArangoDB.

Reads HeroBench's Data/*.json, renders belief nodes (pumpkinspice.corpus),
embeds each via an OpenAI-compatible /v1/embeddings endpoint, and upserts them
into the belief_nodes collection using the READ-WRITE loader user.

Auth: $ARANGO_LOADER_USER / $ARANGO_LOADER_PASSWORD (rw, build-side) -- never the
agent's read-only creds and never the Arango root password. Idempotent: upserts
by _key.

Run:
  uv run --extra arango python scripts/seed_corpus_arango.py \
      --data-dir ~/git/HeroBench/Virtual_Environment/Data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from arango import ArangoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pumpkinspice.corpus import CorpusNode, load_corpus
from pumpkinspice.embeddings import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL

DEFAULT_DATA_DIR = Path.home() / "git/HeroBench/Virtual_Environment/Data"


def embed_batch(client: httpx.Client, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.post("/v1/embeddings", json={"model": model, "input": texts})
    resp.raise_for_status()
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [[float(x) for x in d["embedding"]] for d in data]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--url", default=os.environ.get("ARANGO_URL", "http://localhost:8529"))
    ap.add_argument("--db", default="herobench_kg")
    ap.add_argument("--collection", default="belief_nodes")
    ap.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    user = os.environ.get("ARANGO_LOADER_USER")
    password = os.environ.get("ARANGO_LOADER_PASSWORD")
    if not user or not password:
        print(
            "error: $ARANGO_LOADER_USER/$ARANGO_LOADER_PASSWORD not set "
            "(run scripts/bootstrap_arango.py first).",
            file=sys.stderr,
        )
        return 1
    if not args.data_dir.exists():
        print(f"error: data dir not found: {args.data_dir}", file=sys.stderr)
        return 1

    nodes: list[CorpusNode] = load_corpus(args.data_dir)
    print(f"rendered {len(nodes)} belief nodes from {args.data_dir}")

    embed_client = httpx.Client(base_url=args.embed_url.rstrip("/"), timeout=120.0)
    docs: list[dict] = []
    for start in range(0, len(nodes), args.batch_size):
        batch = nodes[start : start + args.batch_size]
        vectors = embed_batch(embed_client, args.embed_model, [n.text for n in batch])
        for node, vec in zip(batch, vectors, strict=True):
            docs.append(
                {
                    # _key charset is restricted, so sanitize -- but keep the
                    # ORIGINAL id as a field for cross-backend id parity
                    # (pgvector returns "mechanic:crafting_xp"; so must arango).
                    "_key": node.id.replace(":", "_"),
                    "id": node.id,
                    "text": node.text,
                    "metadata": node.metadata,
                    "embedding": vec,
                }
            )
        print(f"  embedded {min(start + args.batch_size, len(nodes))}/{len(nodes)}")

    db = ArangoClient(hosts=args.url).db(args.db, username=user, password=password)
    coll = db.collection(args.collection)
    coll.import_bulk(docs, on_duplicate="replace")
    print(f"upserted {len(docs)} nodes; {args.collection} now has {coll.count()} docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
