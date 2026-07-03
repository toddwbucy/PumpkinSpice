# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Stochastic N-trial batches (web)**: Setup gains a Trials count + Temperature; with
  Trials > 1, `POST /api/trials` runs N trials of one config sequentially, **resetting
  the HeroBench character between each** (`reset_herobench_character`, pure REST
  delete+create -- verified the REST delete now clears the inventory hash, so no redis
  needed), tagging each run `batch:<id>` + `seed:<i>`. A new Reports -> **Batches** view
  aggregates per batch (completed / stopped / errored / avg steps). Critically, trials
  use a **real stochastic sampler** (`_stochastic_sampler`: top_k=0, top_p=0.95, temp,
  seed=i): the decoder's GREEDY default pins top_k=1 (which collapses to one token and
  makes temperature inert) and seed=0, so without this every "re-run" was byte-identical.
  seed=i keeps each trial per-seed reproducible. Each trial is tagged `seed:<i>` +
  `temp:<t>` (the full sampler recipe; top_k/top_p are constants), and a **Stop batch**
  control (`POST /api/trials/{id}/stop`, button in the World tab) aborts the whole batch.
  A **Seed field** on the single-run launch reproduces any trial: `Trials=1, Seed=7,
  Temp=0.7` re-runs that exact stochastic trajectory (blank seed = greedy as before), so
  a winning seed found in a batch can be re-run and studied turn by turn.

- **Agentic Chat with MCP tools (web platform Phase 3a)**: Chat is now an MCP host.
  An in-UI server manager (Settings tab) stores stdio MCP servers
  (`web/mcp_servers.py`, `captures/mcp_servers.json`, `GET/POST/DELETE
  /api/mcp/servers`). When any server is enabled, `/api/chat` runs an agentic
  tool-calling loop (`web/mcp_host.py` + the `mcp` SDK, a new optional extra): it
  spawns the enabled servers per request, exposes their tools to the model
  (namespaced `server__tool`, OpenAI function-calling -- verified qwen3.5 returns
  `tool_calls` through LMStudio), executes tool_calls via the right MCP session,
  feeds results back, and loops (capped). The Chat UI shows each tool call + result.
  Verified end to end (qwen3.5 -> `everything__get-sum(17,25)` -> 42 -> answer).
  Chat-only -- the benchmark agent never gains tools (the fairness firewall). The
  serve command gains `--extra mcp`; the LMStudio endpoint is now env-configurable
  (`PUMPKINSPICE_LMSTUDIO_URL`, default localhost).
- **Model page (web platform Phase 2)**: a persistent app-wide "model under test" +
  decode defaults (`web/settings.py`, `captures/settings.json`). The Model landing
  lists LMStudio models, sets/swaps the model under test, and warm-loads it
  (`POST /api/model/load` -> a JIT load via a tiny request); a decode-defaults form
  (temperature, max_tokens cap, history window) persists via
  `GET/POST /api/model/settings`. Chat and every benchmark run apply these settings
  as defaults, so the chosen model + driving params are configured in one place
  (empty/zero = use the loaded model / unbounded / full history).
- **Web platform restructure (Phase 1b)**: the flat tabs become a nested IA --
  top-level `Model / Chat / HeroBench`, with HeroBench holding `Setup / World /
  Runs / Reports`. The run composer moved to a dedicated **Setup** page (launch ->
  hands off to World); **World** is now a pure live viewer (reattaches to the running
  run); **Reports** is a new tab with a `[ Runs table | Leaderboard ]` toggle --
  the runs table has inline-editable label/tags, the leaderboard ranks models by
  success rate + best/avg steps. See `pumpkinspice-web-PRD.md`.
- **Reporting (web platform Phase 1a)**: a persistent SQLite run registry
  (`reporting.py`, `captures/results.db`) -- a finished web run is recorded with its
  config + computed metrics (from `analyze`) + operator label/tags/notes. Endpoints
  `GET /api/reports/runs` (filter by model/strategy/tag, sort), `GET .../runs/{id}`,
  `POST .../runs/{id}` (annotate), `GET .../leaderboard` (per-model success rate +
  best/avg steps). `pumpkinspice reports-import <captures>` backfills historical
  captures. Separate from the KG; see `pumpkinspice-web-PRD.md`. The benchmark agent
  is unaffected -- reporting is metadata only.
- **Stage 3 (plan + replan-on-surprise)**: a `replan` prompt strategy
  (`prompt_replan.py`). Like Stage 2 it commits a plan on the shared CoT, but the plan
  is LIVING: `observe` updates it whenever the model emits a new `## Plan` (not only
  turn 0), and the prompt invites a keep-or-revise decision after a failed action. This
  is the plan-backbone + reactive-adaptivity combination the Stage-1/2 result argued
  for. `analyze` gains a `replans` (rplan) column counting plan rewrites; config
  `configs/copper_dagger_replan.toml`.
- **Stop-on-goal**: the loop ends as soon as the goal is reached (`goal_item` /
  `goal_level` in `[run]`), so `steps` is steps-to-completion, not always `max_turns`.
  Wired through the CLI and web loop builders; the calibration configs set
  `goal_item = "copper_dagger"`.
- **Move-then-craft** position rule in the CoT/plan/replan prompts: the model must
  stand EXACTLY on the resource/workshop tile to gather/craft and move there first,
  cutting the "one tile short" convergence-cost failures.
- **Recipe book** (`kg.recipe_book` + `scripts/seed_recipe_book.py`): a build-side,
  fully-flattened crafting plan per craftable item -- the multi-hop dependency walk
  (`dagger -> copper -> copper_ore -> copper_rocks@(x,y)`), rolled-up gather
  quantities, gather/craft locations, and the **required skill levels** for each
  step. Fixes the sandbagged one-hop relational join, which dead-ended at a crafted
  ingredient and never surfaced the ore/location (so the agent wandered, not knowing
  where copper came from). The pgvector `relational` mode now appends the precomputed
  `recipe_book` entry for each semantic hit -- a plain runtime lookup; the graph walk
  is build-side (auditable, and conventional RAG, NOT HADES structural retrieval).
- **Reflexion + ReAct chain-of-thought** as the agent's base reasoning method
  (`prompt_default.py`, the `default` strategy). Each turn the model reasons in three
  labeled steps -- Reflect (diagnose why the last action failed, in game terms) ->
  Thought (name the subgoal + the single legal action) -> Action -- then acts. The
  structure is fixed and game-agnostic; content comes from task + world state +
  retrieval, so it generalizes across HeroBench tasks (and, swapping the action
  grammar + corpus, beyond). It replaces the old loose "decide the best next action"
  prompt and is the shared base the `plan` strategy now builds on.
- Per-turn **token throughput**: the decoder captures `usage` (prompt/completion
  tokens) into `Turn.prompt_tokens`/`completion_tokens`; `analyze` adds `gen_tk`
  (avg completion tokens/turn) and `tok/s` columns, making decode latency
  interpretable (big prompt vs long generation vs slow hardware). The comparison
  table also gains a leading `run` column so same-model strategy comparisons (e.g.
  reactive vs plan) are legible, not just cross-model sweeps.
- Planning-ablation **Stage 2 (plan-only)**: a new `plan` prompt strategy
  (`pumpkinspice.plugins.prompt_plan:PlanningPromptBuilder`). The model commits to
  a numbered plan on turn 0; that plan is parsed out via an optional, duck-typed
  `observe(raw)` hook on the loop, held FIXED for the run (no re-planning -- that is
  Stage 3), and shown back each turn while the model executes toward it. The plan is
  a captured artifact (`Turn.plan`). Run it via `configs/copper_dagger_plan.toml`
  (same task/retrieval/sweep as Stage 1, only `prompt = "plan"`) and compare. The
  World tab gains a **Strategy** selector (reactive / plan) and renders the committed
  plan; the decoder HTTP timeout default is raised to 600s (a reasoning model's
  plan turn can think for minutes).
- The active model is now ambient: PumpkinSpice polls LMStudio's currently-loaded
  model (`GET /api/decoder/loaded`) and shows it in a header badge
  (`model · quant · ctx`) -- it does not select or load models (that is LMStudio's
  job). The Playground chats with the loaded model (no model selector); the World
  tab composes a run from **retrieval × task** (no config picker), launched via
  inline options (`POST /api/runs {retrieval, task, max_turns}`). The decoder model
  is recorded per turn for cross-model provenance.
- Planning-ablation Stage 1 tooling: per-turn captures now record the decoder
  `model` (provenance for cross-model analysis); `pumpkinspice analyze <captures>`
  computes outcome metrics (success via `--goal-item`/`--goal-level`, steps,
  failed/no-op actions, wasted moves/revisits, level/xp progress) and prints a
  cross-model comparison; `pumpkinspice sweep -c <cfg> -m a,b,c` runs one config
  across models and compares -- each model id takes an optional `:N` per-model
  max_tokens cap (omitted = unbounded), so rambling non-reasoning models can be
  capped for speed while reasoning models stay uncapped (never truncated mid-think). `configs/copper_dagger.toml` is the calibrated,
  completable multi-step task (goal only -- the reactive baseline). The agent
  configs' `max_turns` are raised to 50 (HeroBench tasks are long-horizon).
- Web frontend: a Vite + React + TypeScript SPA (`frontend/`, pumpkin-spice theme)
  on a FastAPI JSON/SSE backend (`src/pumpkinspice/web/`, `web` extra). Three tabs:
  Playground (backend-agnostic streaming chat -- pick LMStudio/Ollama/vLLM by
  base_url, since all are OpenAI-compatible), Runs (launch a HeroBench config and
  watch turns stream live via SSE), and Captures (browse per-turn JSONL). Launch
  with `pumpkinspice serve` (serves the built SPA + API on one port, default 8077;
  `frontend/` dev via `pnpm dev` proxying /api). Decoder-agnostic: the UI talks to
  our decoder abstraction, not LMStudio's API, so swapping backends is a base_url
  change. UI layer only -- it does not change any experiment invariant. Handles
  reasoning models (Qwen3 etc.): `/api/chat` relays `reasoning_content` separately
  from `content`, and the Playground renders the chain-of-thought in a muted block
  plus the final answer (a content-only parser would show a blank reply).
- World tab: a live HeroBench world viewer. Renders the tile map (color-coded by
  content, via a `GET /api/world/map` proxy of HeroBench `/maps`), the player's
  position, stats (level/HP/xp/gold), the action + outcome, and the model's
  reasoning trace -- all driven by the run's SSE stream, with a turn scrubber and
  follow-live. The harness now captures the reasoning trace per turn: the
  `lmstudio` decoder records `reasoning_content` (`Turn.reasoning`), improving
  capture fidelity (the thinking is planning data for weaver-analysis) as well as
  feeding the viewer.
- Empty-content guard (harness): the agent loop warns loudly and records
  `decoder_empty` in the capture when the decoder returns no content -- a reasoning
  model cut off mid-thought would otherwise silently `rest` every turn. The
  `lmstudio` decoder maps `content: null` to `""`.
- Output is no longer capped: `max_tokens` is the generated-reply cap (not the
  262144 context window, which is set at model load). The `lmstudio` decoder now
  OMITS `max_tokens` when unset (generate until EOS), the agent run configs leave
  it uncapped, and the web chat / Playground default to no cap (0 = unlimited) --
  so reasoning models get the room to finish thinking and answer. (vLLM defaults to
  a tiny cap when omitted; set an explicit value there.)
- Microkernel core: entry-point plugin discovery (`kernel.py`), structural
  `Protocol` contracts (`contracts.py`), and the conventional RAG agent loop
  (`loop.py`).
- Plugin slots `decoder`, `retrieval`, `world`, `prompt`, `capture`, each an
  `importlib.metadata` entry-point group.
- Built-in plugins: decoders `lmstudio` (live) and `echo` (offline); retrieval
  `null` and `pgvector`; world `herobench` (REST) and `mock`; prompt `default`;
  capture `jsonl`.
- `pumpkinspice` CLI: `plugins`, `run`, and stubs for `parity` / `transport`.
- Build-side Postgres provisioning (`scripts/bootstrap_pg.py`): dedicated
  `herobench_kg` database, `kg.belief_nodes` table with an HNSW cosine index,
  and least-privilege roles `ps_loader` (rw) and `ps_agent_ro` (read-only),
  with scoped DSNs written to a gitignored `.env.local`.
- Centralized logging, HTTP connection retries on network plugins, and the
  project quality gates (ruff, mypy strict, pytest + coverage, pre-commit, CI).
- Corpus pipeline: `pumpkinspice.corpus` renders the HeroBench encyclopedia
  (items, monsters, resources, map locations) into belief nodes, and
  `scripts/seed_corpus.py` embeds and upserts them into `kg.belief_nodes` via the
  read-write loader role. First real-retrieval run validated end to end (live
  LMStudio decoder + pgvector search over 302 seeded nodes).

- ArangoDB retrieval arm (2nd ablation backend): `retrieval_arango` plugin (exact
  brute-force cosine in AQL, plain vector search -- no HADES machinery, no
  roproxy), `scripts/bootstrap_arango.py` (database, collection, and least-
  privilege `ps_loader`/`ps_agent_ro` users with default access "none" to every
  other database), and `scripts/seed_corpus_arango.py`. Validated end to end
  (live decoder + arango retrieval + live HeroBench world over 302 seeded docs).

- pgvector "semantic + relational" mode (the `relational = true` toggle): after
  the cosine top-k, it joins the normalized HeroBench schema (`kg.items`,
  `kg.item_craft`, `kg.craft_ingredients`, `kg.sources`, `kg.locations`) to append
  each item's recipe ingredients, the monsters/resources that yield them, and
  where those are found -- structure a vector search alone cannot traverse. New
  `pumpkinspice.relational` extractor (pure, tested) and `scripts/seed_relational_pg.py`.
  Capture `backend` is "pgvector+relational" so the ablation arm is distinguishable.

- In-context turn history: the agent loop feeds its own prior turns (action +
  outcome + position) back into the prompt as conventional working memory
  (`history_window`, default 0 = full history; models run at max context ~200k).
  NOT persisted and NOT written back -- a DB note-taking tool is deliberately
  excluded as that is Agent 1's autonomic memory (the experiment's variable).

- Decoder-parity gate (spec section 4): `pumpkinspice parity --config` greedy-decodes
  fixed prompts via LMStudio, self-checks reproducibility, records the token
  stream + model environment (quantization, arch, context from `/api/v0/models`)
  as an artifact; `pumpkinspice parity --compare A B` diffs two artifacts
  token/text-wise (LMStudio vs the SPU side). Token-ID comparison when LMStudio
  exposes logprobs, else a text fallback (a strong proxy under greedy decode).
- Transport micro-benchmark (spec section 5): `pumpkinspice transport --config`
  reports the round-trip latency distribution (percentiles, not just the mean)
  for a pure `/v1/models` ping and a minimal `max_tokens=1` decode. Confirmed
  transport is negligible (ping p50 ~0.6ms over the LAN).

### Fixed
- **Composer runs now stop on goal.** Web Setup-launched runs never set `goal_item`
  (only the CLI configs did), so they ran to `max_turns` and looped after completing
  the objective. Setup gained a "Goal item (stop on craft)" field, threaded through
  `RunRequest -> _inline_config -> cfg.run["goal_item"]` into the loop's existing
  goal-stop. Also added a cooperative Stop control (`AgentLoop.play(should_stop=...)`,
  a per-run `threading.Event`, `POST /api/runs/{id}/stop`, and a "Stop run" button in
  the World tab) that ends a run cleanly after its current turn (status `stopped`).

- Completion now means the goal item was CRAFTED this run (count rose above the start
  baseline), not merely present. A reset HeroBench character carries residual inventory
  (its `delete` does not clear the `character:<name>:inventory` Redis hash -- a key-name
  bug), so "item present" read as an instant false completion -- a first sweep showed
  3/4 models "winning" in 1 turn. Both the loop's stop-on-goal and `analyze`'s success
  check now compare against the run-start count. (Build-side reset must also
  `redis-cli -n 15 DEL character:<name>:inventory` for a truly clean start.)
- Failed actions now surface the server's REASON to the agent, not just a status
  code. HeroBench returns craft-precondition failures as HTTP 500 with the detail in
  the body (`missing_items`, wrong-tile, skill-too-low); the world plugin previously
  recorded `error = "HTTP 500"` and dropped the body, so the CoT's Reflect step was
  starved (it gathered 11 ore, hit a 500, and had no idea it needed 48). The plugin
  now extracts a concise reason (e.g. "missing items {copper_ore: 37}") into
  `outcome.error` while retaining the full body in `outcome.data`. The recipe-book
  lookup is also capped to the top `recipe_top_n` (3) item hits to cut prompt bloat.
- World tab stopped updating after the first turn on slow runs: per-turn decode is
  20-70s (the prompt grows with full in-context history, plus uncapped output on a
  24B BF16 model), so the SSE sat idle between turns and the UI looked frozen, and a
  page refresh orphaned the run. Fixes: the run-events SSE sends a `: keepalive`
  heartbeat every 2s; the World tab reattaches to an in-progress run on mount
  (survives refresh) via an abortable stream; and it shows a "decoding turn N…
  (~Xs/turn)" liveness indicator.
- pgvector retrieval passed the query embedding as `double precision[]`, which
  has no `<=>` operator against a `vector` column. The vector is now sent as a
  text literal cast with `%s::vector`; covered by a regression test.
- HeroBench world client sent every action body as a JSON object, but endpoints
  with a single scalar Body param (`gathering`) expect the bare value (`1`), not
  `{"quantity": 1}` -- FastAPI returned 422. Single-scalar verbs now send the bare
  value; multi-param verbs (move, craft) still send an object. Regression-tested.
- Prompt now states HeroBench's movement rule (moves must be to an adjacent tile,
  Chebyshev <= 1) and that `gather` takes only a quantity. The agent navigates
  step-by-step instead of repeating failed long-distance moves.

[Unreleased]: https://github.com/toddwbucy/pumpkinspice/commits/main
