# PumpkinSpice

A microkernel RAG-over-HeroBench harness. It is **Agent 2**, the conventional
RAG-over-MCP *control* for the Benchero 2.0 experiment, and a general
model-baselining harness against [HeroBench](../HeroBench).

See `pumpkinspice-Spec.md` for the authoritative spec and `CLAUDE.md` for the
working constraints (OPSEC airgap, decoder parity, fairness).

## Architecture

A small **microkernel** wires five swappable **plugin** slots, discovered via
`importlib.metadata` entry points:

| Slot | Contract | Built-in plugins |
|------|----------|------------------|
| `decoder`   | `complete(prompt, sampler) -> str`             | `lmstudio`, `echo` |
| `retrieval` | `retrieve(query, top_k) -> RetrievalResult`    | `null`, `pgvector`, `arango` |
| `world`     | `get_state()`, `act(action)`                   | `herobench`, `mock` |
| `prompt`    | `query_for(...)`, `build(...) -> str`          | `default` |
| `capture`   | `record(turn)`, `close()`                      | `jsonl` |

The kernel never imports a concrete backend; it resolves `(group, name)` and
constructs the plugin with its config subsection. The retrieval ablation
(`pgvector` -> `arango` -> `hades`, last and only as a flagged non-control run)
is therefore a config swap, not a rewrite.

**Airgap:** retrieval is plain top-k vector search only. It must never delegate
to the HADES CLI's hybrid/rerank/structural retrieval -- HADES is build-side
(seeding the KG), never runtime. A scoped read-only DB user enforces this.

## Quickstart

```bash
uv sync                                              # core + dev tooling
uv run pumpkinspice plugins
uv run pumpkinspice run --config configs/offline.toml   # no external services
```

Quality gates (also enforced in CI):

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
uv run pre-commit install   # run the gates on every commit
```

Real run (needs LMStudio + a seeded pgvector table + HeroBench + a scoped
read-only `$PUMPKINSPICE_PG_DSN`):

```bash
uv sync --extra pgvector
uv run --extra pgvector python scripts/bootstrap_pg.py   # provision DB + scoped roles (admin)
set -a; . .env.local; set +a                             # load the scoped DSNs
uv run --extra pgvector pumpkinspice run --config configs/lmstudio_pgvector.toml
```

## Web frontend

A Vite + React SPA (`frontend/`) on a FastAPI SSE backend, with **Playground**
(backend-agnostic streaming chat — works against any OpenAI-compatible server:
LMStudio, Ollama, vLLM), **World** (live HeroBench map + player stats + reasoning
trace), **Runs** (launch a config, watch turns stream live), and **Captures** tabs.

```bash
uv sync --extra web
cd frontend && pnpm install && pnpm build    # build the SPA once (serve picks up frontend/dist)

# Serve the API + built SPA. Include the DB extras so DB-backed runs (pgvector /
# arango) can build; serve auto-loads scoped creds from .env.local.
uv run --extra pgvector --extra arango --extra web pumpkinspice serve   # http://127.0.0.1:8077

# Frontend dev with hot reload (proxies /api -> :8077), no manual refresh:
cd frontend && pnpm dev                      # http://127.0.0.1:5273
```

## Backing services

- Decoder: LMStudio at `http://192.168.0.203:1234` (OpenAI-compatible; LAN host).
- pgvector: Postgres at `localhost:5432`.
- KG: ArangoDB at `localhost:8529` (plain HTTP, needs auth).

Models connect with their **own scoped, least-privilege** credentials, never the
root passwords. See `CLAUDE.md` for the provisioning posture.

## Status

Built to release-quality engineering standards (ruff, mypy strict, pytest +
coverage, pre-commit, CI). Working end to end: the microkernel; the plugin
contracts; the offline run path; real `lmstudio` decoder, `herobench` world, and
**two retrieval arms** (`pgvector` and `arango`), each with a scoped-role
bootstrap, a corpus seeder, and a validated real run. Not yet built: the
decoder-parity gate (spec s4), the transport micro-benchmark (s5), the `hades`
retrieval arm (build-side, last), and the planned web frontend.
