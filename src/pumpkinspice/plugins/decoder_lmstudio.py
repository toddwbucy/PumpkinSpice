"""Decoder backed by LMStudio's OpenAI-compatible endpoint (spec section 3).

Loads the same GGUF the WeaverTools SPU loads; the decoder-parity gate (spec
section 4) pins sampler + tokenizer settings. Sampler defaults here are greedy
(temperature 0, top-k 1, no repeat penalty, fixed seed) so a scored run is
reproducible -- pass overrides via the per-call ``sampler`` or config.

Endpoint defaults to the known LAN host (192.168.0.203:1234). Note: that is a
LAN hop, not localhost -- relevant to the transport micro-benchmark (section 5),
which frames the LMStudio path as "localhost".
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "http://192.168.0.203:1234"

# Greedy defaults for reproducible / parity-aligned decoding.
GREEDY: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "repeat_penalty": 1.0,
    "seed": 0,
}


class LMStudioDecoder:
    name = "lmstudio"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.model = config.get("model")  # LMStudio uses the loaded model if omitted
        # max_tokens caps the GENERATED reply, not the context window (262144, set
        # at model load). Default 0 = UNBOUNDED, and that is the correct default for
        # a benchmark: a reasoning model must finish its (variable, growing) thinking
        # before it emits the action, so ANY fixed cap can cut it off mid-thought ->
        # empty action -> a corrupted turn. The cost is that a non-reasoning model
        # with no natural stop can ramble past the answer (slow, but still correct --
        # the action JSON is first; the per-turn time is captured as a metric). Cap
        # it per-config (e.g. 256) for a specific non-reasoning model if you want
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
        self._defaults = {**GREEDY, **config.get("sampler", {})}
        # Retry transient connection failures (LAN host may briefly blip).
        transport = httpx.HTTPTransport(retries=int(config.get("retries", 2)))
        self._client = httpx.Client(
            base_url=self.base_url, timeout=self.timeout, transport=transport
        )

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
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
        # Capture the chain-of-thought (reasoning models put it in reasoning_content,
        # separate from the answer) so the harness can record it per turn.
        reasoning = message.get("reasoning_content")
        self.last_reasoning = reasoning if isinstance(reasoning, str) else ""
        # A reasoning model that did not finish thinking returns content: null. Map
        # null -> "" so the empty case is detectable, not the string "None".
        content = message.get("content")
        return content if isinstance(content, str) else ""
