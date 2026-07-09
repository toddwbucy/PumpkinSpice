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

import logging
from typing import Any, ClassVar

import httpx

log = logging.getLogger(__name__)


class OpenAICompatDecoder:
    # Overridden per backend. The base defaults are deliberately inert -- this
    # class is a shared base, not a registered plugin.
    name: str = "openai_compat"
    DEFAULT_BASE_URL: ClassVar[str] = "http://127.0.0.1:8000"
    GREEDY: ClassVar[dict[str, Any]] = {"temperature": 0, "top_p": 1, "seed": 0}
    require_model: ClassVar[bool] = False
    # Whether this backend honors vLLM-style chat_template_kwargs (the transport for the
    # `enable_thinking` no-think flag). LMStudio silently ignores it, so setting
    # enable_thinking there would keep internal CoT on while the config says off.
    supports_chat_template_kwargs: ClassVar[bool] = False

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
        # The ACTUAL request body sent by the most recent complete(), minus the prompt
        # messages: the effective sampler (incl. seed), max_tokens, model, and any extra_body
        # exactly as merged onto the wire. The loop records this as the capture's decode
        # provenance (the experiment's IV) so the record cannot disagree with what was sent.
        self.last_request: dict[str, Any] = {}
        self._defaults = {**self.GREEDY, **config.get("sampler", {})}
        # Request-body passthrough for backend-specific fields (merged into the chat
        # payload). `enable_thinking` is the v2 reasoning-location IV: set False to disable
        # a Qwen3-style model's internal CoT via vLLM's chat_template_kwargs, so reasoning
        # is externalized into bounded harness steps (pair with a small max_tokens). Left
        # unset -> the model's default (internal-CoT baseline arm).
        self.extra_body: dict[str, Any] = dict(config.get("extra_body", {}))
        enable_thinking = config.get("enable_thinking")
        if enable_thinking is not None:
            # The dedicated knob WINS over any chat_template_kwargs.enable_thinking already
            # in extra_body (a plain assign, not setdefault -- else a stray baseline value
            # could silently invert the IV).
            ctk = dict(self.extra_body.get("chat_template_kwargs", {}))
            ctk["enable_thinking"] = bool(enable_thinking)
            self.extra_body["chat_template_kwargs"] = ctk
            if not self.supports_chat_template_kwargs:
                log.warning(
                    "%s decoder: enable_thinking=%s is sent via chat_template_kwargs, which "
                    "this backend is not known to honor -- it may silently keep internal CoT "
                    "ON. Verify the reasoning field of the captures.",
                    self.name,
                    enable_thinking,
                )
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
        self.last_request = {}
        s = {**self._defaults, **(sampler or {})}
        # extra_body FIRST so reserved fields win over it: the greedy/parity sampler (`s`,
        # incl. the per-call sampler=), messages, max_tokens, and model must not be
        # clobbered by a stray extra_body key (e.g. a temperature in extra_body must not
        # override the decode-parity sampler). extra_body then only supplies non-reserved
        # backend fields (chat_template_kwargs, guided_choice, ...).
        payload: dict[str, Any] = {
            **self.extra_body,
            "messages": [{"role": "user", "content": prompt}],
            **s,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.model:
            payload["model"] = self.model
        # Snapshot the request AS SENT (minus the prompt) for capture provenance: reserved
        # keys that clobbered extra_body are reflected as they went on the wire, and
        # max_tokens/model are included. Set before the POST so a raising request still
        # records what was attempted.
        self.last_request = {k: v for k, v in payload.items() if k != "messages"}
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
