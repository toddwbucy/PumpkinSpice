"""Shared base for decoders backed by an OpenAI-compatible /v1/chat/completions
endpoint (spec section 3).

Both LMStudio (llama.cpp, the pinned-GGUF path for scored runs) and vLLM (the
baselining/floor-test path) speak the same wire protocol, so the HTTP call,
sampler merge, token-usage capture, and chain-of-thought extraction all live here
once. Subclasses differ only in four class attributes:

* ``name``            -- the plugin name (also the capture provenance).
* ``DEFAULT_BASE_URL``-- where the server listens by default.
* ``GREEDY``          -- the greedy sampler in *that backend's* dialect. llama.cpp
                         spells it ``repeat_penalty`` + ``top_k: 1``; vLLM spells
                         it ``repetition_penalty`` + ``top_k: -1``. Same intent,
                         different vocabulary -- so it cannot be shared.
* ``require_model``   -- vLLM 400s without a ``model`` matching the served name;
                         LMStudio falls back to whatever model is loaded.

The Decoder Protocol (contracts.py) is structural, so a subclass that only sets
these attributes conforms with no extra boilerplate.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx


class OpenAICompatDecoder:
    # Overridden per backend. The base defaults are deliberately inert -- this
    # class is a shared base, not a registered plugin.
    name: str = "openai_compat"
    DEFAULT_BASE_URL: ClassVar[str] = "http://127.0.0.1:8000"
    GREEDY: ClassVar[dict[str, Any]] = {"temperature": 0, "top_p": 1, "seed": 0}
    require_model: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.base_url = str(config.get("base_url", self.DEFAULT_BASE_URL)).rstrip("/")
        self.model = config.get("model")
        # vLLM's /v1/chat/completions requires a `model` that matches the served
        # name; there is no "use the loaded model" fallback as in LMStudio. Fail
        # fast at construction with a clear message rather than on the first turn.
        if self.require_model and not self.model:
            raise ValueError(
                f"{self.name} decoder requires a 'model' in config "
                "(the served model name); it has no loaded-model fallback"
            )
        # max_tokens caps the GENERATED reply, not the context window. Default 0 =
        # UNBOUNDED, and that is the correct default for a benchmark: a reasoning
        # model must finish its (variable, growing) thinking before it emits the
        # action, so ANY fixed cap can cut it off mid-thought -> empty action ->
        # a corrupted turn. The cost is that a non-reasoning model with no natural
        # stop can ramble past the answer (slow, but still correct -- the action
        # JSON is first; the per-turn time is captured as a metric). Cap it
        # per-config (e.g. 256) for a specific non-reasoning model if you want
        # speed and know it won't truncate a real answer.
        self.max_tokens = int(config.get("max_tokens", 0))
        # HTTP read timeout. Non-streaming, so it must exceed the WHOLE generation:
        # a reasoning model writing a plan (Stage 2, turn 0) can think 5000+ tokens
        # -> minutes. Default 600s; a single timeout aborts the run, so err generous.
        self.timeout = float(config.get("timeout", 600.0))
        # The chain-of-thought from the most recent complete() (reasoning models
        # return it separately from the answer). Captured per turn by the loop.
        self.last_reasoning: str = ""
        # Token counts from the most recent complete() (prompt_tokens, completion_tokens).
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._defaults = {**self.GREEDY, **config.get("sampler", {})}
        # Retry transient connection failures (the server host may briefly blip).
        transport = httpx.HTTPTransport(retries=int(config.get("retries", 2)))
        self._client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=transport
        )

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        # Clear per-call state up front: if the request raises (e.g. a 400 when the
        # prompt exceeds a small-context model's window), the caller must see cleared
        # reasoning/usage for THIS turn, not the previous turn's stale values (which
        # would double-count its tokens and mis-attribute its chain-of-thought).
        self.last_reasoning = ""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        s = {**self._defaults, **(sampler or {})}
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            **s,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.model:
            payload["model"] = self.model
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        message = body["choices"][0]["message"]
        # Token throughput: prompt/completion token counts from `usage` make the
        # per-turn decode latency interpretable (a slow turn = big prompt? long
        # generation? -- not just a wall-clock number). Derived tok/s is computed
        # by the loop, which owns the wall-clock.
        usage = body.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
        # Capture the chain-of-thought so the harness can record it per turn. Field
        # name varies by model family: Qwen-style models use `reasoning_content`,
        # gpt-oss (harmony) uses `reasoning`. Prefer the former, fall back to the
        # latter -- without this, gpt-oss's thinking is silently dropped (its tokens
        # still count in usage.completion_tokens, so a turn looks expensive with no
        # visible reason). vLLM populates `reasoning_content` when run with a
        # --reasoning-parser, so the same handling covers both backends.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        self.last_reasoning = reasoning if isinstance(reasoning, str) else ""
        # A reasoning model that did not finish thinking returns content: null. Map
        # null -> "" so the empty case is detectable, not the string "None".
        content = message.get("content")
        return content if isinstance(content, str) else ""
