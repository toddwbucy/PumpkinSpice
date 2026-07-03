# PumpkinSpice Web Platform - PRD / Spec

Status: DRAFT (2026-06-28), awaiting operator ratification. Editorial: ASCII, no em-dashes.

This document specifies the evolution of the PumpkinSpice web UI from a single-benchmark
prototype (Playground / World / Runs / Captures tabs) into a multi-benchmark, local model-testing
and benchmarking platform with persistent reporting. It is the seed of the fall-2026 public release.

It COMPLEMENTS, and does not override, `pumpkinspice-Spec.md` (the experiment apparatus spec) and the
hard constraints in `CLAUDE.md`. Where this PRD and the experiment spec disagree about the benchmark
agent or the fairness contract, the experiment spec wins.

--------------------------------------------------------------------------------

## 1. Vision

PumpkinSpice is a local web platform for testing and benchmarking models. From one UI an operator
can: configure/load the model under test, chat with it (agentically, with tools + memory + file
attachments), set up a benchmark run, watch it play, and produce a persistent, tagged report of how
the model performed. HeroBench is the first benchmark; the structure generalizes so a second
benchmark is a self-contained module added beside it.

## 2. Goals

- One UI for the full loop: pick the model -> chat -> configure a benchmark run -> watch it ->
  report on it.
- Persistent, taggable run records with computed metrics (reporting that survives server restarts).
- A benchmark-extensible information architecture (HeroBench first; a new benchmark is a module with
  the same Setup / World / Runs / Reports shape).
- An agentic chat for interacting with the model under test: MCP tools, MCP memory, file attachments.
- Operator control of which model is loaded (load/swap via LMStudio) plus app-wide decode defaults.
- Release-quality throughout: typed, linted, tested, documented.

## 3. Non-goals (for this PRD)

- Changing the experiment or fairness contract (owned by `pumpkinspice-Spec.md`).
- Giving the benchmark AGENT MCP tools or autonomic memory. The benchmark agent stays a controlled
  conventional-RAG control (see section 5). MCP and memory live in Chat only.
- Multi-user accounts, cloud hosting, or remote auth. The platform is local-first; "login" means
  opening the local app. (Auth may be revisited for the public release; out of scope here.)

## 4. Information architecture

```
PumpkinSpice
+- Model         landing. Load/swap the model under test (via LMStudio) + app-wide decode defaults
+- Chat          agentic chat with that model: MCP tools + MCP memory + file attachments
+- HeroBench     a benchmark section (future: sibling benchmark sections beside it)
   +- Setup      its main page: task / level / objective, database & retrieval, strategy -> Launch
   +- World      live tile map + player stats + reasoning trace while a run plays
   +- Runs       the live / recent run feed (in-flight + just-finished)
   +- Reports    persistent, SQLite-backed; [ Runs table | Leaderboard ] toggle; tag / label / notes
```

The current flat tabs map in: Playground -> Chat; World/Runs stay under HeroBench; Captures graduates
into Reports (raw captures remain browsable but are now indexed by a run registry).

## 5. Core architectural principle: Chat vs Benchmark agent

Chat and the HeroBench agent are DIFFERENT consumers of the same model, and must stay separate:

- Chat gets all the bells: MCP tools, MCP memory, file attachments, free-form interaction.
- The benchmark agent stays the controlled experiment: plain top-k vector search over belief nodes,
  no MCP tools, no autonomic memory, least-privilege read-only DB role. This is the whole fairness
  contract (CLAUDE.md hard constraints; `pumpkinspice-Spec.md` sections 6-8).

If MCP tooling or memory ever leaks into the benchmark run loop, the control is no longer a
conventional-RAG control and the comparison is invalid. Implementation must keep the Chat stack and
the benchmark run loop as separate code paths sharing only the decoder (the model under test).

## 6. Decisions (ratified by operator, 2026-06-28)

- Run-record store: SQLite, a single `captures/results.db`, separate from the KG Postgres.
- Reports view: a `[ Runs table | Leaderboard ]` toggle; runs are taggable/labelable with notes.
- Model page: PumpkinSpice load/swaps the model under test via LMStudio (reverses the earlier
  "poll only, leave loading to LMStudio" stance) and sets app-wide decode defaults.

## 7. Data model (reporting)

A run record is written to `captures/results.db` when a run finishes (or is stopped). Sketch:

```
runs(
  id            text primary key,    -- run id
  benchmark     text,                -- "herobench"
  model         text,                -- model under test
  strategy      text,                -- reactive | plan | replan (the prompt slot)
  retrieval     text,                -- pgvector | pgvector+relational | arango | null
  task          text,                -- the objective text
  goal          text,                -- goal_item / goal_level (machine-checkable success)
  max_turns     integer,
  started_at    text,                -- ISO timestamp
  finished_at   text,
  status        text,                -- done | error | stopped
  -- computed metrics (from analyze): steps, success, failed_actions, no_ops, revisits,
  -- replans, level_delta, xp_delta, avg_decode_ms, avg_gen_tokens, decode_tok_s, action_counts
  metrics       text,                -- JSON blob of the RunMetrics
  capture_path  text,                -- pointer to the raw per-turn JSONL
  label         text,                -- operator display label
  tags          text,                -- JSON array of tags
  notes         text                 -- operator free text
)
```

Metrics come from the existing `analyze` module (RunMetrics); the registry persists them so Reports
does not recompute on every view. Raw per-turn data stays in `captures/<...>.jsonl`, referenced by
`capture_path`.

## 8. Phased delivery

### Phase 1 - IA restructure + Reports (priority; grounded)

Mostly wiring + persistence; the metrics already exist (`analyze`) and captures are the raw data.

Backend:
- A persistent run registry over `captures/results.db` (create table, upsert on run finish).
- On run completion, compute metrics via `analyze` and write the run record.
- Endpoints: list runs (filter by benchmark/model/strategy/tag, sort by any metric); get one run;
  update tags/label/notes; leaderboard aggregate (per-model / per-strategy best + average).

Frontend:
- Nested nav: Model / Chat / HeroBench{ Setup, World, Runs, Reports }.
- HeroBench Setup page: reorganize the current World composer into a dedicated config page
  (task/objective/level, retrieval/database, strategy, max_turns, goal) -> Launch.
- Reports tab: `[ Runs table | Leaderboard ]` toggle.
  - Runs table: one row per run, its metrics, sortable/filterable, inline tag/label/notes editing,
    "compare selected" -> the analyze comparison view.
  - Leaderboard: per-model/strategy ranking (best & average steps-to-completion, success rate).
- Captures browsing remains reachable (raw JSONL) but indexed by the registry.

### Phase 2 - Model page (load/swap + decode defaults)

- List LMStudio models; show which is loaded; pick/swap the model under test (JIT-loads on next use).
  Confirm quantization / loaded context length.
- App-wide decode defaults applied to chat + benchmark-run defaults: temperature/sampler,
  max_tokens cap, history window.
- Explicit unload/eject and load-progress are a follow-on (likely need the LMStudio SDK or `lms`
  CLI rather than the plain REST API); swap-by-JIT ships first.

### Phase 3 - Agentic chat (MCP + memory + file attachments)

Most novel and highest-risk; do it once the shell is in place. Mini-spec resolved 2026-06-28; split
into 3a then 3b. CONSTRAINT throughout: all of this is Chat-only -- the benchmark agent loop never
gains MCP/memory/tools, and chat-attachment chunks live in a session-scoped store, never the
HeroBench KG (the fairness firewall).

Phase 3a - MCP host + in-UI server manager + tool loop (no new infra):
- In-UI server manager: add / edit / enable MCP servers from a Settings page, persisted like the
  model settings (a server = name + launch command + args + enabled). PumpkinSpice spawns enabled
  servers over stdio (the `mcp` Python SDK, a new optional extra) and aggregates their tools
  (namespaced by server).
- The /api/chat endpoint becomes an agentic tool-calling loop: send the aggregated tools to LMStudio
  (OpenAI-style function-calling; capable models like qwen3.5 support it), execute any tool_calls via
  the right MCP session, append results, loop until the model answers. The chat stream surfaces
  tool-call activity (tool, args, result) alongside reasoning + content.
- "Memory" is just one of the configured MCP servers (a memory MCP); nothing special-cased.

Phase 3b - file-attach RAG (pluggable: pgvector / ArangoDB / Neo4j):
- Paste text into the input = inline (always; just text in the message).
- The paperclip = attach a file -> chunk + embed + retrieve into a configured RAG store, on each turn.
  The button is GATED: it appears only when a RAG backend is configured; otherwise the feature is not
  available.
- A pluggable `RagStore` (embed -> store -> retrieve) over three backends: pgvector and ArangoDB
  (drivers exist from the benchmark retrieval) and Neo4j (NET-NEW: provision a Neo4j instance + a
  5.x vector index + a scoped least-privilege user, matching the pg/arango bootstrap pattern).
- Attachment chunks go to a session-scoped collection, isolated from the HeroBench KG.

## 9. Open questions / deferred decisions

- MCP (Phase 3a): RESOLVED -- in-UI server manager (not a static file); servers via stdio. Open:
  per-model tool-calling support (verify qwen3.5 etc. through LMStudio); streaming vs non-streaming
  tool loop for v1.
- File attachments (Phase 3b): RESOLVED -- paste = inline; paperclip = extract-and-RAG, gated on a
  configured RAG backend (pgvector / ArangoDB / Neo4j). Open: which file types beyond text/code (PDF
  extraction?); image/vision support (deferred -- current model under test is text-only).
- Model unload/eject mechanism (Phase 2 follow-on): LMStudio SDK vs `lms` CLI vs skip.
- Reports: export formats (CSV / JSON); run-vs-run comparison UI; whether tags double as
  campaign/experiment grouping.
- Auth/login for the eventual public release (out of scope now).

## 10. Constraints carried over

- Fairness/OPSEC (binding): the benchmark agent stays conventional-RAG, no hades, no MCP, no
  autonomic memory; retrieval is plain top-k vector search via the least-privilege read-only DB role.
  See CLAUDE.md hard constraints and `pumpkinspice-Spec.md` sections 6-8.
- DB isolation: `results.db` is local experiment metadata, separate from the KG databases; holds no
  secrets and no scoped DB credentials.
- Editorial: ASCII only, no em-dashes, in all repo docs.
- Release-quality: ruff + mypy strict + pytest with branch coverage >= 80%, pre-commit, CI.

## 11. Status of the underlying harness (context, 2026-06-28)

The benchmark engine these tabs drive is already validated end to end: HeroBench is completable by
the conventional-RAG agent (copper_dagger crafted in 14-16 steps by ministral-3-14b, mistral-24b,
qwen3.6-27b; gemma-4-26b fails by fixating on gathering). Three planning strategies exist as the
`prompt` slot (reactive / plan / replan), with stop-on-goal, the recipe-book retrieval, move-then-
craft, and informative failure feedback. `analyze` computes the metrics this PRD persists. The work
here is the platform/reporting layer on top of that engine.
