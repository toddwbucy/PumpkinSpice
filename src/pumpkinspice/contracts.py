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
    # Decode provenance: the effective sampler (incl. seed) and extra_body actually sent
    # this turn. This is the experiment's IV record -- e.g. the reasoning-location arm is
    # decode["extra_body"]["chat_template_kwargs"]["enable_thinking"] -- so runs are
    # groupable by their sampler/no-think settings post-hoc (empty if the decoder does not
    # expose them, e.g. the mock/echo decoders).
    decode: dict[str, Any] = field(default_factory=dict)


# --- Plugin Protocols -------------------------------------------------------


@runtime_checkable
class Decoder(Protocol):
    """LMStudio-style text decoder. Greedy/sampler settings come from config so
    the decoder-parity gate (spec section 4) can pin them exactly."""

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
