# PumpkinSpice - Status Report for WeaverTools

Date: 2026-06-28. Editorial: ASCII, no em-dashes. Audience: the WeaverTools / Benchero 2.0 team.

This report summarizes PumpkinSpice (Agent 2, the conventional RAG-over-MCP control) two days into
its build. It covers the architecture, the baselining results to date, what those results imply for
the WeaverTools comparison, and a work log. It contains no WeaverTools internals; PumpkinSpice holds
none by construction and stays OPSEC-clean.

--------------------------------------------------------------------------------

## 1. Executive summary

PumpkinSpice is a HeroBench model-baselining harness and the conventional-RAG control for Benchero
2.0. In two days it went from a single spec file to a working, release-quality harness that an agent
can use to play HeroBench end to end, plus a web platform with persistent reporting.

The headline finding from baselining is honest and useful: once the agent is given COMPLETE task
knowledge (a precomputed "recipe book") plus informative action feedback, even the simplest reactive
agent completes the calibration task in near-optimal steps, and the planning architecture (commit-a-
plan, replan-on-surprise) adds only marginal gains. The dominant lever was KNOWLEDGE COMPLETENESS,
not planning sophistication. This reframes where WeaverTools' autonomic memory has to win: not on
"having a plan," but on tasks whose solution cannot be pre-flattened and retrieved (see section 5).

Status: the harness is validated and the control completes the calibration task. These are results
from the GENERAL-BASELINING use of the harness, not a scored Benchero 2.0 run (that needs the locked
model / task set / scorer matched to Agent 1; it is future work).

## 2. What PumpkinSpice is

Two uses, one harness:

1. Benchero 2.0 control arm (its origin). Here it is Agent 2, the conventional RAG-over-MCP control
   measured against Agent 1 (the WeaverTools autonomic harness). For a scored run the fairness
   contract is binding: decoder, KG content, task set, and scorer must all be identical to Agent 1;
   only the tooling (conventional RAG vs autonomic) and the system prompt differ.
2. General model baselining (broader). The same harness baselines models across other experiments;
   the fairness contract is not the constraint there, only a clean reusable RAG-vs-HeroBench loop.

Hard constraints honored (these define the experiment):

- OPSEC-clean by construction: the repo holds none of the WeaverTools harness or SPU internals, so
  it can be built in its own open session.
- Conventional RAG only: the MODEL decides when to retrieve; the harness does not maintain a world
  model and does not autonomically surface memory. That contrast with Agent 1 is the whole point.
- hades is build-side only (airgap): we use hades with admin creds to BUILD the knowledge graph; the
  harness/model runtime never reaches hades or its hybrid/rerank/structural retrieval. The agent's
  only retrieval is a thin tool doing plain top-k vector search via a scoped read-only DB user.
- Least-privilege DB: each model/experiment gets its own scoped DB user on its own database; the
  agent runtime never uses root.

## 3. Architecture

Microkernel / plugin pattern. A small stable core (the RAG agent loop + plugin contracts +
config/orchestration) wires swappable plugins for the parts that vary across the ablation. Plugins
are discovered by entry points and selected by config; the kernel stays ignorant of any concrete
backend. The five pluggable slots:

- decoder    - LMStudio over its OpenAI-compatible endpoint, loading the exact GGUF the SPU loads.
- retrieval  - plain top-k vector search over belief nodes, behind one interface. Backends built and
               validated: pgvector (semantic), pgvector+relational (semantic + a precomputed recipe
               chain), and ArangoDB (KG). NOT hades; plain SQL/AQL by hand.
- world      - the HeroBench REST surface (move/fight/gather/craft/equip/rest + get-state).
- prompt     - the system prompt + prompt construction. Three planning strategies live here (below).
- capture    - a per-turn record (rendered prompt, raw output, retrieval + latency, action, outcome)
               shaped to align with Agent 1's per-turn record. A labeled corpus, not just a log.

Plus two gates that run before scored work: a decoder-parity gate (decode a fixed prompt at greedy
through both LMStudio and the SPU GGUF backend; confirm token streams match) and a transport
micro-benchmark (decoder round-trip latency distribution). Both implemented and validated live;
greedy decode is reproducible, transport ping p50 ~0.6ms. NOTE for the scored run: the parity gate
revealed mistral-small-24b loads at 8192 context by default (arch max 32768), not ~200k; verify the
loaded context per model before relying on long full history.

Reasoning method (the prompt slot). The base reasoning method is a Reflexion + ReAct chain-of-
thought: each turn the model Reflects (diagnose why the last action failed, in game terms), Thinks
(name the subgoal and the single legal action), then Acts. The structure is fixed and game-agnostic;
content comes from the task + world state + retrieval, so it generalizes across HeroBench tasks. The
three planning strategies are the ablation's independent variable, all sharing that CoT:

- reactive (Stage 1): the CoT applied fresh each turn, no committed plan.
- plan (Stage 2): commit a plan on turn 0 and execute it; no revision.
- replan (Stage 3): a living plan that revises when the world contradicts it.

The recipe book (a key result, section 5). Conventional one-hop relational retrieval dead-ended at a
CRAFTED ingredient (e.g. "needs 6x copper") and never reached the ore or its location, so the agent
could not navigate. We added a build-side step that walks the recipe graph ONCE and writes a fully
flattened plan per item (gather quantities, smelt steps, workshop and resource coordinates, required
skill levels) into a SQLite-free relational table; the runtime retrieval is then a plain lookup. This
is conventional RAG (a lookup over our own table), NOT hades structural retrieval, and the agent
never runs the graph walk.

Web platform. A Vite/React SPA on a FastAPI SSE backend, recently restructured into a nested
information architecture: top-level Model / Chat / HeroBench, with HeroBench holding Setup (configure
and launch a run), World (live tile map + stats + reasoning trace), Runs (live feed), and Reports.
Reports is backed by a persistent SQLite run registry (separate from the KG) with a runs table
(taggable/labelable) and a leaderboard. A future agentic Chat (MCP tools + memory + file
attachments) is specified but is Chat-only; the benchmark agent stays the controlled conventional-RAG
control. The full product spec is `pumpkinspice-web-PRD.md`.

## 4. Results to date (general baselining)

Calibration task: craft a copper dagger. A completable multi-step chain across locations: gather
copper ore at the copper rocks tile, smelt copper at the mining workshop, craft the dagger at the
weaponcrafting workshop. Chosen because it exercises the full plan -> execute -> fail -> adapt ->
succeed cycle and has a machine-checkable success (the dagger in inventory, crafted this run).

Cross-model sweep (one run each, clean character reset between models; recipe book + CoT +
move-then-craft + stop-on-goal):

```
model                 steps   completed   failed-actions   revisits   tok/s
ministral-3-14b        14       yes             0              2       18.7
mistral-small-24b      15       yes             1              3        7.2
qwen3.6-27b            16       yes             0              6       40.1
gemma-4-26b-a4b        50       NO              0             47        8.1
```

- Three of four models complete the task in 14-16 steps (near the ~12-action floor). Conventional
  RAG, properly equipped, is genuinely capable here.
- Capability is not size: the smallest model (ministral, 14B) tied for best; the larger gemma (26B)
  failed.
- gemma's failure is a multi-step-TRANSITION failure: it did 47 gather actions one at a time and
  never advanced to smelting (it only ever reached the resource tile). It had the knowledge; it
  could not manage the hand-off between steps.

Planning-stage ablation (mistral, same equipment, the only variable is the strategy):

```
strategy            steps   completed   failed-actions   replans
replan  (Stage 3)    14       yes             0             8
plan    (Stage 2)    15       yes             0             0
reactive(Stage 1)    15       yes             1             0
```

All three complete; Stage 3 is marginally best (one fewer step, zero fails, and it revised its plan
8 times adaptively). The gaps are small.

## 5. The key finding and what it means for WeaverTools

The story is in how the gaps SHRANK. Before the recipe book and informative feedback, the same
strategies differed wildly on this task:

- A loose reactive prompt thrashed: ~43 failed actions out of 50 turns, no completion. This is the
  conventional-RAG failure mode that motivated WeaverTools in the first place, reproduced cleanly.
- A proper CoT cut failures to ~8 but went cautious (it stopped attempting the hard action and
  wandered) - low failures, still no completion.
- Adding the recipe book let the agent navigate the whole chain and gather the exact materials, but
  it stalled one tile short because action failures returned only an opaque status code.
- Adding informative failure feedback ("wrong tile, go to (1,5)"; "missing 37 copper_ore") closed
  the loop: the first end-to-end completion.

Once knowledge was complete and feedback was informative, even the reactive baseline completes in 15
steps, and the planning layer adds little. So:

The dominant lever was KNOWLEDGE COMPLETENESS, not planning architecture.

For the WeaverTools comparison this is a sharp, honest claim, and it sharpens the hypothesis rather
than weakening it. For a task whose full solution can be pre-computed and retrieved (a recipe book),
conventional RAG suffices and a plan buys little. WeaverTools' autonomic co-resident memory therefore
has to demonstrate its edge where that pre-computation is NOT possible:

- fog-of-war discovery, where the agent must accumulate state the world only reveals over time;
- long-horizon tasks whose relevant knowledge cannot be flattened into a static document ahead of
  time, and must be surfaced from what the agent has done;
- situations where the live world contradicts static knowledge and the agent must maintain and
  re-surface a corrected world model.

That is precisely the regime autonomic memory is for. The control's result tells the experiment where
to point the camera: choose tasks the recipe book cannot solve, or the comparison will under-measure
WeaverTools' advantage.

A related observation: the reactive-vs-rigid axis is a false choice. Before the knowledge fix, a
committed-but-flawed plan (Stage 2, no revision) was WORSE than adaptive reactive CoT, because it
could not self-correct; reactive CoT avoided failures but lacked goal-direction. You need plan goal-
direction AND reactive adaptivity together - which is Stage 3, and ultimately autonomic memory.

## 6. Engineering status

- Release-quality bar: ruff format + lint, mypy strict, pytest with branch coverage. Currently 75
  tests pass at ~82% coverage; lint and type-check clean. The frontend type-checks and builds.
- Both retrieval backends (pgvector, arango) provisioned with least-privilege scoped roles, seeded,
  and validated end to end against the live world.
- Parity gate and transport gate implemented and validated.
- Per-turn captures double as a labeled corpus; an analyze layer computes the metrics (success,
  steps, failed actions, revisits, replans, token throughput) and a cross-model comparison.
- A persistent SQLite run registry records finished runs with metrics + operator tags/labels; the
  Reports tab serves a runs table and a leaderboard. Tonight's runs are imported and visible.

## 7. Not yet done / caveats

- These are general-baselining results, NOT a scored Benchero 2.0 run. A scored run requires the
  locked model, task set, and scorer matched to Agent 1, and the shared ArangoDB KG content Agent 1
  sees. The harness implements its side of the parity contract; the scored matrix is the experiment
  document's to fix.
- Per-model loaded context must be verified (the 8192-vs-32768 finding) before relying on long full
  history in a scored run.
- The web platform is Phase 1 of a larger redesign (model load/swap and an agentic MCP chat are
  specified but unbuilt). None of that touches the benchmark agent, which stays controlled.
- Single-run-per-cell so far; a scored comparison would want repeated runs per cell for variance.

## 8. Two-day work log

Day 1 (2026-06-26 to 06-27): greenfield spec to working harness. Built the microkernel (kernel,
plugin contracts, the RAG loop, CLI, config), the pgvector and ArangoDB retrieval backends with
least-privilege bootstrap + corpus seeding, the LMStudio decoder, the HeroBench world client (on the
Redis backend, isolated to logical db 15 to protect other experiments' data), the decoder-parity and
transport gates, and a first web frontend (chat playground + a live World viewer + runs + captures).
Handled reasoning models (chain-of-thought in a separate field) and the max_tokens-vs-context
distinction.

Day 2 (2026-06-28): the planning ablation and the platform. Added the three planning strategies
(reactive / plan / replan) and the analyze metrics layer; replaced the loose prompt with a proper
Reflexion + ReAct CoT; built the recipe book (the knowledge fix) and informative action feedback,
which produced the first end-to-end completion; added stop-on-goal, the move-then-craft rule, and the
cross-model sweep with clean character resets. Caught and fixed two measurement bugs (a residual-
inventory false positive and a nested-response false negative) that would have made the result tables
lie. Then specified the web platform redesign (PRD) and shipped Phase 1: a persistent SQLite reporting
registry and the restructured nested UI (Model / Chat / HeroBench{Setup, World, Runs, Reports}) with a
taggable runs table and a model leaderboard.

## 9. Appendix - pointers

- `pumpkinspice-Spec.md` - the apparatus/experiment spec (authoritative for the fairness contract).
- `pumpkinspice-web-PRD.md` - the web platform product spec (phases, decisions, open questions).
- `CHANGELOG.md` - the detailed change history.
- `captures/results.db` + `captures/*.jsonl` - the run registry and raw per-turn captures.
