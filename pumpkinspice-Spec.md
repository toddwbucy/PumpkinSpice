# Benchero 2.0 - Baseline Repo Spec (PumpkinSpice)

Status: DRAFT (2026-06-27), awaiting operator ratification. Editorial: ASCII, no em-dashes.

This specifies PumpkinSpice, the external sibling repository that implements Agent 2, the conventional RAG-over-MCP control for Benchero 2.0. It is a separate, independent Python repository, coded and run in its own session. It holds none of the WeaverTools harness or SPU internals, so it stays out of the WeaverTools conformance graph and is OPSEC-clean by construction. The apparatus it plugs into is `benchero-2.0_apparatus.md` (this directory); the experiment it serves is `../experiment/benchero-2.0_experiment.md`, where it is H3's external comparison arm.

---

## 1. Purpose and role

Agent 2, codenamed PumpkinSpice, is the stack a competent local developer would actually build to play HeroBench with retrieval-augmented generation: LMStudio for the decoder, a knowledge graph behind an MCP retrieval tool, and a conventional RAG agent loop. It is the control against which the WeaverTools autonomic harness (Agent 1) is measured.

The comparison is fair by construction. The decoder is the same (verified, section 4), the knowledge-graph content is the same, the HeroBench task set is the same, and the scorer is the same. The only things that differ are the tooling (conventional RAG-over-MCP versus autonomic co-resident memory) and the system prompt (typical versus small). A planning-quality difference is therefore attributable to those two, which is the whole point of the experiment.

## 2. Components (all Python)

- **HeroBench client** - the REST calls (`move`/`fight`/`gather`/`craft`/`equip` plus get-state).
- **KG-over-MCP retrieval** - a thin MCP server exposing the shared ArangoDB knowledge graph as a retrieval tool (vector search over the belief nodes), plus the agent-side MCP client. This is the conventional retrieval mechanism, the deliberate contrast with WeaverTools' write-first autonomic surfacing.
- **Decoder** - LMStudio serving the GGUF over its OpenAI-compatible endpoint.
- **The RAG agent loop** - get world state, retrieve via the MCP tool, build a typical RAG prompt, call the decoder, parse the action, act via the HeroBench client.
- **Capture** - a per-turn record comparable to the apparatus's weaver-trace capture (planning data plus retrieval latency), so `weaver-analysis` can align the two agents after the fact.
- **The decoder-parity check and the transport micro-benchmark** (sections 4 and 5).

## 3. Interfaces (the seam, fixed by the apparatus)

- **Decoder:** LMStudio's OpenAI-compatible endpoint, loading the exact same GGUF file the WeaverTools SPU loads.
- **Retrieval:** the shared ArangoDB knowledge graph (the seeded HeroBench encyclopedia, the same content Agent 1 sees), read through this repo's own MCP server.
- **World and scoring:** the HeroBench REST API and its `scoring_pipeline` (the labeler), the same as Agent 1.

## 4. The decoder-parity contract (shared, do not redefine)

The same GGUF on two llama.cpp instances is not the same decoder until proven. Before any scored run, run the decoder-parity gate defined in apparatus section 8: decode a fixed prompt at greedy (temperature 0, top-k 1, no repeat penalty, a fixed seed) through both LMStudio and the SPU GGUF backend, and confirm the token streams match. Record the LMStudio build and its bundled llama.cpp version, the SPU's `llama-cpp-sys-2` version, the sampler settings, and the tokenizer settings. A mismatch is a version or sampler skew to pin before any run. This repo implements its side of that one contract. It does not invent its own.

## 5. The transport micro-benchmark

Measure decoder round-trip latency over localhost (LMStudio's endpoint) against the WeaverTools unix-socket path on a fixed prompt set, to bound and rule out the transport as a differentiator (expected negligible). Report the distribution, not just the mean.

## 6. Fairness constraints (no sandbagging, no strawman)

- A typical, competent RAG system prompt, what a good practitioner would write.
- Sensible retrieval: good tool descriptions, a reasonable top-k, query construction a competent developer would use.
- No WeaverTools tooling: no write-first world-model maintenance, no autonomic surfacing, no tiered always-injected memory. This is conventional RAG, where the model decides when to retrieve and the harness does not maintain a world model.
- The target is the strongest fair version of the conventional pattern, so that a WeaverTools win is a real win and a WeaverTools loss is honest.

## 7. Capture (comparable, and training-ready by design)

Per turn, record the rendered prompt, the raw model output, the retrieval calls and their latency, the action taken, and the HeroBench outcome, in a form `weaver-analysis` can align with the WeaverTools per-turn record. The HeroBench scorer supplies the outcome labels. The capture is a labeled corpus, not just a log.

## 8. Scope

**In:** the conventional RAG agent, the MCP retrieval server, the LMStudio decoder integration, the HeroBench play loop, the capture, the parity check, the transport benchmark.

**Out:** anything from the WeaverTools harness (the autonomic loop, the SPU, the tiered memory). The H3 part-two ablations belong to WeaverTools, not here. Model selection and the run matrix are the experiment document's to fix, not this repo's.

## 9. Deliverables

- The Python repository, its own repo and its own CI.
- A documented run: the decoder-parity gate passing, then the RAG agent playing the locked HeroBench task set under the locked model, producing the comparable per-turn capture for `weaver-analysis`.

## 10. Pointers

- The apparatus contract: `benchero-2.0_apparatus.md` (sections 6 to 8, and the decoder-parity gate).
- The experiment: `../experiment/benchero-2.0_experiment.md` (H3, the control's role, and the run matrix in open question 1).
- HeroBench: `~/git/HeroBench` (the REST server and `scoring_pipeline.py`).
- The baseline design rationale: `[[project_benchero_baseline_design]]` (agent memory).
