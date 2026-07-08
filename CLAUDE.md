# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

First vertical slice exists and runs, held to release-quality standards (ruff, mypy strict, pytest
+ branch coverage >=80%, pre-commit, GitHub Actions CI): the microkernel (`kernel.py`), plugin
contracts (`contracts.py`), the RAG loop (`loop.py`), CLI (`cli.py`), and built-in plugins. An
offline run (`echo`/`null`/`mock`) and a live `lmstudio` decoder run both work end to end and write
JSONL captures. The pgvector backend is provisioned and seeded: `scripts/bootstrap_pg.py` creates
the `herobench_kg` DB, `kg.belief_nodes` table, and least-privilege roles; `scripts/seed_corpus.py`
fills it with 305 belief nodes from the HeroBench encyclopedia. Both retrieval arms (`pgvector` and
`arango`) are provisioned, seeded, and validated end to end against the live HeroBench world. The
decoder-parity gate (`pumpkinspice parity`, spec s4) and transport micro-benchmark
(`pumpkinspice transport`, spec s5) are implemented and validated live (greedy decode is
reproducible; transport ping p50 ~0.6ms). A **web frontend** exists: a Vite+React SPA on a FastAPI
SSE backend (`src/pumpkinspice/web/`, `web` extra; `pumpkinspice serve`), with Playground
(backend-agnostic streaming chat), World (live HeroBench map + player stats + reasoning trace), Runs
(live SSE), and Captures tabs. There is **no hades retrieval arm** - the model never uses hades (see
hard constraints). The frontend is a polishable v1 (the seed of the fall-2026 release).

Note: the parity gate records the loaded model's `quantization` and `loaded_context_length`. It
revealed `mistral-small-24b` loads at 8192 context (arch max 32768), NOT the intended ~200k -- verify
per model before relying on long full history.

This is a **professional, release-quality** project (operator directive): new code ships typed,
linted, tested, and documented. The airgap and least-privilege guarantees are release-blocking
invariants, not nice-to-haves.

**`pumpkinspice-Spec.md` remains the authoritative source.** When this file or the code disagrees
with the spec, the spec wins (and this file should be updated).

## Commands (uv)

`uv` lives at `~/.local/bin/uv` (add to PATH). `src/` layout, editable install, `uv.lock` committed.

```bash
uv sync                                              # core + dev tooling (dev group is default)
uv run pumpkinspice plugins                          # list discovered plugins per slot
uv run pumpkinspice run --config configs/offline.toml # offline end-to-end, no externals
uv sync --extra pgvector                             # add the pgvector backend driver
```

Quality gates (all run in CI; `uv run pre-commit install` mirrors them locally):

```bash
uv run ruff format --check .   # formatting   (ruff format . to fix)
uv run ruff check .            # lint          (ruff check --fix . to fix)
uv run mypy                    # strict type-check (src only)
uv run pytest                  # tests + branch coverage, fails under 80%
uv run pytest tests/test_config_cli.py::test_cli_run_offline --no-cov  # single test (skip cov gate)
```

Tests use `httpx.MockTransport` / fakes, so they need no live services. The
pgvector tests `importorskip` and run only with the `pgvector` extra (CI installs it).

Build-side corpus seeding + a real retrieval run (admin/build-side; seeding needs the DBs +
Ollama for embeddings; a live decoder is only needed for the final `run` step):

```bash
uv run --extra pgvector python scripts/bootstrap_pg.py          # provision DB + scoped roles
uv run --extra pgvector python scripts/seed_corpus.py           # embed HeroBench encyclopedia -> kg.belief_nodes (ps_loader, rw)
uv run --extra pgvector python scripts/seed_relational_pg.py    # normalize the schema -> kg.items/item_craft/craft_ingredients/sources/locations
uv run --extra pgvector python scripts/seed_recipe_book.py      # flatten the recipe GRAPH -> kg.recipe_book (full chain + levels + locations)
set -a; . .env.local; set +a                                    # load scoped DSNs
uv run --extra pgvector pumpkinspice run --config configs/real_retrieval.toml  # live decoder + pgvector (ps_agent_ro, ro)
```

`pumpkinspice/corpus.py` is the pure, tested renderer (build-side content prep,
not runtime retrieval); the seeder script does the IO/embedding/upsert. The recipe
book does the multi-hop graph walk build-side so the runtime `relational` retrieval
is a plain lookup (see "recipe book" below); re-run `seed_recipe_book.py` after
`seed_relational_pg.py` whenever the schema tables change.

Planning-ablation strategies (the `prompt` slot, all sharing the Reflexion+ReAct CoT):
`default` = reactive (Stage 1), `plan` = commit-and-execute (Stage 2), `replan` =
plan + replan-on-surprise (Stage 3). Configs: `copper_dagger{,_plan,_replan}.toml`
(all set `goal_item` so the run stops on completion). Compare with `pumpkinspice
analyze` (the `rplan` column counts plan rewrites). `sweep -c <cfg> -m a:N,b` runs a
config across models with per-model `:N` max_tokens caps.

Web frontend (Playground / World / Runs / Captures tabs):

```bash
cd frontend && pnpm install && pnpm build    # build the SPA once
# Serve API + SPA. Include the DB extras so DB-backed runs (pgvector/arango) can
# build their drivers; `serve` auto-loads scoped creds from .env.local.
uv run --extra pgvector --extra arango --extra web --extra mcp pumpkinspice serve   # http://127.0.0.1:8077
cd frontend && pnpm dev                       # hot-reload dev server (proxies /api -> :8077)
```

The canonical serve command includes all three extras -- a plain `uv run pumpkinspice serve` would
sync the DB drivers out, so DB-backed runs would fail to build (now surfaced as a clean 400, not a
500). `serve` reads `.env.local` so the scoped DB creds are present without sourcing it.

Configs select plugins per slot in `[run]` and pass per-plugin settings in `[<slot>]` tables; see
`configs/offline.toml` (offline), `configs/live_decoder_smoke.toml` (real decoder, stub rest),
and `configs/lmstudio_pgvector.toml` (full real-run template).

## What this repo is

PumpkinSpice is a HeroBench model-baselining harness. It plays HeroBench (`~/git/HeroBench`) with
retrieval-augmented generation: an LMStudio decoder + a retrieval backend behind an MCP tool + a
plain RAG agent loop. It has two uses:

1. **Benchero 2.0 control arm (its origin, per `pumpkinspice-Spec.md`).** Here it is **Agent 2**,
   the conventional RAG-over-MCP *control* measured against Agent 1, the WeaverTools autonomic
   harness (a different repo and session). For these scored runs the fairness contract below is
   binding: decoder, KG content, task set, and scorer must all be identical to Agent 1.
2. **General model baselining (operator directive, broader than the spec).** The same harness is
   used to baseline models for evaluation across *other* experiments. Here the fairness contract is
   not the constraint; the goal is a clean, reusable RAG-vs-HeroBench evaluation loop.

When a run is a Benchero 2.0 scored run, treat the constraints below as the experiment's contract:
they are not style preferences, and getting them wrong invalidates the comparison.

## Hard constraints (these define the experiment, do not relax without operator sign-off)

- **OPSEC-clean by construction.** This repo holds *none* of the WeaverTools harness or SPU
  internals, and must not. That is why it can be coded in its own open session. Do not import,
  reference, or reconstruct: the autonomic loop, the SPU, write-first world-model maintenance,
  autonomic surfacing, or tiered always-injected memory. If a task seems to need WeaverTools
  internals, it is out of scope (see spec section 8), stop and confirm.
- **Conventional RAG only.** The *model* decides when to retrieve; the harness does not maintain a
  world model. This deliberate contrast with Agent 1 is the whole point, do not "improve" it into
  autonomic behavior.
- **The `hades` CLI is build-side only (airgap).** We (Claude + operator) use `hades` with admin
  creds to *build* the graph: create the database, apply schema, ingest the HeroBench encyclopedia,
  embed, verify. The harness / model runtime must NEVER reach `hades` or its advanced retrieval
  (hybrid search, reranking, structural graph-embedding retrieval). The agent's only retrieval is
  the thin MCP tool doing **plain top-k vector search** over belief nodes, via a scoped read-only DB
  user. Subtle trap: the MCP server we build must NOT delegate to `hades db query -H hybrid -R
  rerank -S structural` - implement plain vector search by hand. Letting the control reach
  structural/hybrid retrieval would invalidate the comparison.
- **No sandbagging, no strawman.** Target the *strongest fair* version of conventional RAG: a
  competent practitioner's system prompt, good MCP tool descriptions, a sensible top-k, reasonable
  query construction. A WeaverTools win must be a real win; a loss must be honest.
- **Parity is shared, not redefined.** The decoder-parity contract and the seam (decoder /
  retrieval / world+scoring) are fixed by the apparatus. This repo implements *its side* of that
  one contract and does not invent its own (spec sections 3 and 4).
- **Editorial: ASCII only, no em-dashes** in repo docs (the spec follows this; match it).

## Database auth and isolation (hard security rule)

These are shared databases; isolation is mandatory, not optional.

- **Root creds are admin-only.** `ARANGO_PASSWORD` and `POSTGRESQL_PASSWORD` (env vars) hold the
  ArangoDB and Postgres root/superuser passwords. Use them only for provisioning/administration.
  Never print, log, or commit them.
- **The model/agent runtime must NEVER use root.** Each model/experiment gets its OWN login user
  scoped to its OWN database, and must not be able to read or overwrite any other database. The
  agent reads its scoped creds from its own config/env, never the root env vars.
- **Provisioning posture (least privilege):**
  - *Postgres:* per-model role with LOGIN only, no SUPERUSER/CREATEDB/CREATEROLE; grant privileges
    on its own database/schema only; `REVOKE CONNECT` on other databases and from PUBLIC so a role
    cannot reach databases it was not explicitly granted.
  - *ArangoDB:* per-model user with `rw` (or `ro`) on its own database only; default database
    access level "No access", and `none` on `_system` and every other database.

## Architecture (all Python, per spec section 2)

**Structure: microkernel / plugin pattern.** A small stable core (the RAG agent loop + plugin
contracts + config/orchestration) wires swappable plugins for the parts that vary across the
ablation. The plugin seam is the whole point - it is what lets the retrieval-variant ablation be a
config swap, not a rewrite. Pluggable slots: **retrieval backend**, **decoder**, **world client**,
**prompt builder**, **capture sink**. Keep the kernel ignorant of any concrete backend; a plugin is
selected by config and must satisfy only its interface.

Retrieval-variant ablation (all CONVENTIONAL): **pgvector** (pure semantic vector store) ->
**pgvector+relational** (semantic + SQL schema joins) -> **ArangoDB** (KG with semantic structure).
**There is no HADES retrieval arm** - the model never uses hades (see hard constraints). HADES is the
structural/autonomic technology behind Agent 1 (WeaverTools), the comparison target, which is out of
scope for this repo (spec section 8). "Testing with hades" is the experiment-level Agent 2 (this
repo) vs Agent 1 comparison, not a feature here.

The five pieces and how they connect:

1. **HeroBench client** - REST calls `move`/`fight`/`gather`/`craft`/`equip` plus get-state,
   against the HeroBench FastAPI server.
2. **Retrieval over MCP** - a thin MCP *server* exposing a retrieval tool (vector search over
   belief nodes), plus the agent-side MCP *client*. This is the conventional retrieval mechanism.
   The backend is **pluggable behind one retrieval interface** - do not hardcode a backend. Both
   arms are built and validated:
   - **`pgvector`** on Postgres (`localhost:5432`): HNSW cosine, ~60ms/query. Has a `relational =
     true` mode that joins the normalized HeroBench schema (`kg.items/item_craft/craft_ingredients/
     sources/locations`, seeded by `scripts/seed_relational_pg.py`) to append each semantic hit's
     recipe -> ingredients -> sources -> locations. Plain SQL joins (conventional RAG), NOT HADES
     structural retrieval. Capture `backend` becomes "pgvector+relational".
   - **`arango`** on ArangoDB (`localhost:8529`, plain HTTP): exact brute-force cosine in AQL,
     ~3s/query (3.12 vector index is experimental, not used). PumpkinSpice's own direct client,
     *not* the HADES unix-socket roproxy/rwproxy path used elsewhere in `~/git`.
   Both seed via a `ps_loader` (rw) role and query via a `ps_agent_ro` (ro) role; provisioned by
   `scripts/bootstrap_{pg,arango}.py`, seeded by `scripts/seed_corpus{,_arango}.py`.
   For a Benchero 2.0 scored run, the backend and its content must match Agent 1 (use the ArangoDB
   KG there unless the operator says otherwise). The pgvector path and other backends are for the
   general-baselining use. Retrieval is **plain top-k vector search**, hand-built - never delegated
   to the `hades` CLI's hybrid/rerank/structural retrieval (see hard constraints). In practice the
   corpora are seeded by our own `seed_corpus*.py` scripts (plain embeddings + inserts), not hades;
   either way the model only ever does plain vector search over its read-only DB user.
3. **Decoder integration** - LMStudio serving the GGUF over its OpenAI-compatible endpoint.
4. **RAG agent loop** - get world state -> retrieve via MCP tool -> build a typical RAG prompt ->
   call decoder -> parse action -> act via HeroBench client. The prompt includes **in-context turn
   history** (the agent's own prior actions/outcomes; `history_window` config, default 0 = full
   history, since models run at max context ~200k). This is conventional working memory - NOT
   persisted, NOT written back. A DB note-taking / world-model-write tool is deliberately excluded:
   that is Agent 1's autonomic memory and the experiment's independent variable (enforced by the
   read-only `ps_agent_ro` role). See the hard constraints.
5. **Capture** - a per-turn record (rendered prompt, raw model output, retrieval calls + latency,
   action, HeroBench outcome label) shaped so `weaver-analysis` can align it with Agent 1's
   per-turn record. Treat capture as a labeled training corpus, not just a log.

Plus two gates that run *before* scored work:
- **Decoder-parity gate** (spec section 4): decode a fixed prompt at greedy (temp 0, top-k 1, no
  repeat penalty, fixed seed) through both LMStudio and the SPU GGUF backend; confirm token streams
  match. Record LMStudio build + bundled llama.cpp version, the SPU's `llama-cpp-sys-2` version,
  sampler settings, tokenizer settings. A mismatch is a version/sampler skew to pin before any run.
- **Transport micro-benchmark** (spec section 5): decoder round-trip latency over LMStudio's
  localhost endpoint vs the WeaverTools unix-socket path on a fixed prompt set. Report the
  *distribution*, not just the mean.

## The seam (external interfaces, fixed by the apparatus)

- **Decoder:** LMStudio's OpenAI-compatible endpoint, loading the *exact same GGUF* the WeaverTools
  SPU loads.
- **Retrieval:** for Benchero 2.0, the shared ArangoDB knowledge graph (the seeded HeroBench
  encyclopedia, identical content to what Agent 1 sees), read through this repo's own MCP server.
  For general baselining, either retrieval backend (ArangoDB KG or pgvector) is allowed.
- **World and scoring:** the HeroBench REST API and its `scoring_pipeline.py` labeler, identical to
  Agent 1.

The fairness guarantee is that decoder, KG content, task set, and scorer are all identical to
Agent 1; only the *tooling* (conventional RAG vs autonomic) and the *system prompt* (typical vs
small) differ. Do not introduce any other difference.

## External dependencies and pointers

- **HeroBench** lives at `~/git/HeroBench` (present on disk). FastAPI world server, default URL
  `http://127.0.0.1:8000`. Two backends: `Virtual_Environment/FastApi_Redis_Ver` and
  `FastApi_SQLite_Ver`, each started with `fastapi run`. Scoring is `scoring_pipeline.py`. The
  existing `A2_Agent/` there is a reference RAG agent worth reading before building this one.
- **Apparatus** `benchero-2.0_apparatus.md` (sections 6-8 and the parity gate) and **experiment**
  `../experiment/benchero-2.0_experiment.md` (H3, the control's role, the run matrix) are cited by
  the spec but are **not present in this tree yet** - do not assume they exist locally; ask the
  operator for them when their detail is needed.
- **Decoder:** LMStudio at `http://192.168.0.203:1234` (LAN host, OpenAI-compatible endpoint), not
  localhost. Note this for spec section 5's transport benchmark, which assumes a localhost path.
- **Retrieval backends:** ArangoDB at `localhost:8529` (plain HTTP, needs auth) and Postgres/
  pgvector at `localhost:5432`. Both confirmed reachable.
- **Embeddings:** headless **Ollama** at `http://localhost:11434` serving `nomic-embed-text`
  (768-dim), the default for retrieval and both seeders (defined once in
  `pumpkinspice/embeddings.py`). Decoupled from the decoder so no LMStudio GUI is needed; the web
  backend overrides via `PUMPKINSPICE_EMBED_URL` / `PUMPKINSPICE_EMBED_MODEL` (as a unit). It MUST
  match the model that seeded the corpus -- if you change the embedder, re-run BOTH
  `seed_corpus.py` (pgvector) AND `seed_corpus_arango.py` (arango), since query and document
  vectors must share one space per arm (a 768-dim mismatch degrades silently, no error). To catch
  that, the seeders stamp the embed model into each node's metadata and retrieval verifies it on the
  first query (name + vector dimension + mixed-corpus/partial-reseed), failing fast on a mismatch.
  For a known-same-space rename (an embedder whose name differs across servers), set
  `embed_model_check = "warn"` (or `"off"`) in the retrieval config. For scored latency runs set
  `OLLAMA_KEEP_ALIVE=-1` so an idle model reload is not billed into retrieval latency.
- **Model selection and the run matrix** are the experiment document's to fix, not this repo's.

## Out of scope (spec section 8)

Anything from the WeaverTools harness (autonomic loop, SPU, tiered memory); the H3 part-two
ablations; model selection and the run matrix. Deliverables are: this Python repo with its own repo
and CI, and a documented run (parity gate passing, then the RAG agent playing the locked task set
under the locked model, producing the comparable per-turn capture).
