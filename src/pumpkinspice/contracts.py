"""Plugin contracts and shared data types for the PumpkinSpice microkernel.

The kernel depends only on these structural Protocols, never on a concrete
plugin. A plugin satisfies a contract by shape (duck typing), so out-of-tree
plugins need not import this module to be loadable.

Every plugin class is constructed with a single argument: its config dict
(the relevant subsection of the run config). See ``kernel.load_plugin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --- Shared value types -----------------------------------------------------


@dataclass
class BeliefNode:
    """One retrieved unit of evidence (a "belief node" in the KG/vector store)."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    nodes: list[BeliefNode]
    latency_ms: float
    backend: str = ""


@dataclass
class WorldState:
    """Opaque world/character state plus the source it came from."""

    raw: dict[str, Any]
    source: str = ""


@dataclass
class Action:
    """A parsed action the agent intends to take.

    ``kind`` is the verb (move/fight/gather/craft/equip/rest/...); ``args`` the
    verb-specific parameters; ``raw_text`` the model text it was parsed from.
    """

    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class ActionResult:
    ok: bool
    status_code: int
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Turn:
    """The per-turn capture record (spec section 7). Shaped so weaver-analysis
    can align it with the WeaverTools per-turn record. This is a labeled corpus
    row, not just a log line."""

    index: int
    task: str
    world_state: dict[str, Any]
    retrieval: dict[str, Any]
    prompt: str
    raw_output: str
    action: dict[str, Any]
    outcome: dict[str, Any]
    timings_ms: dict[str, float]
    # True when the decoder returned no content (e.g. a reasoning model that ran
    # out of max_tokens mid-thought); the action then defaults to `rest`.
    decoder_empty: bool = False
    # The model's chain-of-thought for this turn (reasoning models only; "" otherwise).
    reasoning: str = ""
    # The decoder model id (provenance for cross-model analysis; "" if not exposed).
    model: str = ""
    # The agent's committed plan, if the prompt strategy maintains one (Stage 2+).
    plan: str = ""
    # Token throughput for this turn's decode (0 if the decoder did not report usage).
    # tok/s is derivable as completion_tokens / (timings_ms["decode"] / 1000).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Decode provenance: the request the decoder ACTUALLY sent this turn, minus the prompt
    # messages -- the effective sampler (incl. seed), max_tokens, model, and any extra_body,
    # exactly as merged onto the wire (so the record cannot disagree with what was sent).
    # This is the experiment's IV record -- e.g. the reasoning-location arm is
    # decode["chat_template_kwargs"]["enable_thinking"] and the length-cap is
    # decode["max_tokens"] -- so runs are groupable by their decode settings post-hoc (empty
    # if the decoder does not expose `last_request`, e.g. the mock/echo decoders).
    decode: dict[str, Any] = field(default_factory=dict)
    # The served model's ENVIRONMENT: precision (operator-declared `quantization`/`dtype`) and
    # the context window it actually loaded at (server-verified). Distinct from `decode` (the
    # request) and `model` (the id): precision is not on the wire, and a silent context
    # downgrade is invisible without recording the served length (cf. the parity gate finding
    # a model loaded at 8192, not the intended ~200k). Precision-sensitive analysis such as the
    # #7/#8 trajectory geometry must be able to condition on it. Empty for decoders that do not
    # expose `model_info` (mock/echo).
    model_info: dict[str, Any] = field(default_factory=dict)
    # The decoder's stop reason for this turn ("stop" = natural end, "length" = hit the token/
    # context cap, "" if not reported). A "length" finish means the reply was TRUNCATED -- for a
    # reasoning task that is a cut-off trace with no final answer, which grades "incorrect" in a
    # LENGTH-correlated way. Recording it lets analysis separate "hit the wall" from "genuinely
    # wrong" instead of conflating them (the confound the floor test must rule out).
    finish_reason: str = ""


# --- Plugin Protocols -------------------------------------------------------


@runtime_checkable
class Decoder(Protocol):
    """LMStudio-style text decoder. Greedy/sampler settings come from config so
    the decoder-parity gate (spec section 4) can pin them exactly.

    The loop also reads these OPTIONAL provenance attributes if present (duck-typed via
    getattr, so a bare Protocol implementation is fine but records less): ``last_request``
    (the request body sent by the last complete(), minus messages -> Turn.decode),
    ``last_reasoning`` (chain-of-thought), ``last_usage`` (prompt/completion token counts),
    and ``model`` (the model id). A decoder that wants full capture provenance should set
    ``last_request`` in complete()."""

    name: str

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str: ...


@runtime_checkable
class Retrieval(Protocol):
    """Conventional retrieval: plain top-k vector search over belief nodes.

    MUST NOT delegate to HADES hybrid/rerank/structural retrieval -- that is
    build-side only and would invalidate the conventional-RAG control.
    """

    name: str

    def retrieve(self, query: str, *, top_k: int) -> RetrievalResult: ...


@runtime_checkable
class World(Protocol):
    """The HeroBench play surface."""

    name: str

    def get_state(self) -> WorldState: ...

    def act(self, action: Action) -> ActionResult: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Renders the typical RAG prompt (a competent practitioner's, not a
    strawman; see fairness constraints)."""

    name: str

    def build(
        self,
        *,
        state: WorldState,
        retrieval: RetrievalResult,
        task: str,
        history: list[Turn],
    ) -> str:
        """Render the prompt. ``history`` is the recent in-context turns (a rolling
        window of the agent's own prior actions/outcomes) -- conventional working
        memory, NOT persisted and NOT a written-back world model."""
        ...

    def query_for(self, *, state: WorldState, task: str) -> str:
        """The retrieval query a conventional RAG agent would construct."""
        ...


@runtime_checkable
class Capture(Protocol):
    name: str

    def record(self, turn: Turn) -> None: ...

    def close(self) -> None: ...
