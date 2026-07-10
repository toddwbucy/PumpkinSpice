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
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from pumpkinspice.model_probe import fetch_model_entry

log = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    """One decode's full result, returned by value instead of stashed on the decoder.

    ``complete()`` keeps its single-call ``last_*`` snapshot for the sequential agent loop,
    but that snapshot is shared mutable state -- unsafe under the concurrent decoding the
    batched MATH path uses. ``complete_many()`` returns one of these per prompt so nothing is
    clobbered across threads. Fields mirror the ``last_*`` snapshot: ``content`` is the answer,
    ``request`` is the decode provenance (the payload minus the messages)."""

    content: str
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request: dict[str, Any] = field(default_factory=dict)


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
        # Served-model precision for capture provenance (Turn.model_info). vLLM's OpenAI
        # endpoint does not report precision, so it is OPERATOR-DECLARED here: `quantization`
        # (e.g. "none" for an unquantized checkpoint, or "Q4_K_M" for a GGUF) and `dtype`
        # (e.g. "bfloat16"). The served context length is server-verified in `model_info`, not
        # declared. The LMStudio subclass overrides discovery to read both from its native
        # endpoint, so a pinned-GGUF scored run records quant without declaring it.
        self.quantization = config.get("quantization")
        self.dtype = config.get("dtype") or config.get("precision")
        # Server-verified half of model_info (context length, and for LMStudio quant/arch),
        # memoized ONLY once discovery succeeds -- so a transient blip at first access is
        # retried, not permanently cached as "no server data" (which would strip the
        # served-window proof from the whole capture). None = not yet discovered.
        self._discovered: dict[str, Any] | None = None
        # The chain-of-thought from the most recent complete() (reasoning models
        # return it separately from the answer). Captured per turn by the loop.
        self.last_reasoning: str = ""
        # The stop reason from the most recent complete() ("stop"/"length"/...); "length" means
        # the reply was truncated at the cap. Captured per turn so a cut-off (answerless) trace
        # is distinguishable from a wrong answer.
        self.last_finish_reason: str = ""
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

    def _build_payload(
        self, prompt: str, sampler: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the chat payload and the provenance snapshot (payload minus messages)."""
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
        request = {k: v for k, v in payload.items() if k != "messages"}
        return payload, request

    def _send(self, payload: dict[str, Any], request: dict[str, Any]) -> DecodeResult:
        """POST one chat payload and parse it into a DecodeResult. Touches no ``self``
        state, so it is safe to call concurrently (httpx.Client is thread-safe)."""
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        message = choice["message"]
        # Stop reason: "length" = truncated at the token/context cap (a reasoning trace cut off
        # before its answer). Recorded so this is not misread as a wrong answer.
        finish_reason = str(choice.get("finish_reason") or "")
        # Token throughput: prompt/completion counts make decode latency interpretable
        # (big prompt? long generation? -- not just a wall-clock number).
        usage = body.get("usage") or {}
        # Chain-of-thought field varies by model family: Qwen-style uses `reasoning_content`,
        # gpt-oss (harmony) uses `reasoning`. Prefer the former, fall back to the latter --
        # else gpt-oss's thinking is silently dropped (its tokens still count in usage). vLLM
        # populates `reasoning_content` when run with a --reasoning-parser.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        # A reasoning model that did not finish thinking returns content: null -> "" so the
        # empty case is detectable, not the string "None".
        content = message.get("content")
        return DecodeResult(
            content=content if isinstance(content, str) else "",
            reasoning=reasoning if isinstance(reasoning, str) else "",
            finish_reason=finish_reason,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            request=request,
        )

    def decode_one(self, prompt: str, sampler: dict[str, Any] | None = None) -> DecodeResult:
        """Decode one prompt, returning the result by value with NO ``self`` mutation -- the
        thread-safe unit the batched MATH runner fans out concurrently (httpx.Client is
        thread-safe; vLLM batches the in-flight requests server-side). ``complete()`` is the
        sequential path that additionally stashes the ``last_*`` snapshot for the agent loop."""
        payload, request = self._build_payload(prompt, sampler)
        return self._send(payload, request)

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        # Clear per-call state up front: if the request raises (e.g. a 400 when the
        # prompt exceeds a small-context model's window), the caller must see cleared
        # reasoning/usage for THIS turn, not the previous turn's stale values (which
        # would double-count its tokens and mis-attribute its chain-of-thought).
        self.last_reasoning = ""
        self.last_finish_reason = ""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.last_request = {}
        payload, request = self._build_payload(prompt, sampler)
        # Record the attempt BEFORE the POST so a raising request still shows what was sent.
        self.last_request = request
        res = self._send(payload, request)
        self.last_usage = {
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
        }
        self.last_finish_reason = res.finish_reason
        self.last_reasoning = res.reasoning
        return res.content

    @property
    def model_info(self) -> dict[str, Any]:
        """The served model's environment (precision + context window), for capture
        provenance (Turn.model_info). Combines OPERATOR-DECLARED precision (config
        ``quantization``/``dtype``) with SERVER-VERIFIED fields (the context length the
        server actually loaded; server values win). Best-effort: a server with no model
        metadata just yields the declared fields (never raises, never blocks a run). The
        server half is cached only once discovery SUCCEEDS, so a transient blip at the first
        access -- e.g. the MATH runner snapshotting this before the first decode -- is retried,
        not memoized as permanently absent."""
        declared: dict[str, Any] = {"backend": self.name}
        if self.model:
            declared["model"] = self.model
        if self.quantization is not None:
            declared["quantization"] = self.quantization
        if self.dtype is not None:
            declared["dtype"] = self.dtype
        if self._discovered is not None:
            disc = self._discovered
        else:
            disc = self._discover_model_info()
            if disc:  # cache only a non-empty (successful) discovery; retry a blip next access
                self._discovered = disc
        # server-verified fields last so a real loaded context/quant wins over anything declared
        return {**declared, **disc}

    def _discover_model_info(self) -> dict[str, Any]:
        """Best-effort query of the served context window from the OpenAI-compatible
        ``/v1/models`` endpoint. vLLM reports ``max_model_len``; record it as
        ``served_context_length`` so the capture proves the run got the intended window
        (cf. the parity gate finding a model silently loaded at 8192, not ~200k). Returns
        ``{}`` when the endpoint has no match/field; the shared probe uses a short timeout
        and swallows errors. LMStudio's native quant/context live on a different endpoint
        (see the subclass override)."""
        m = fetch_model_entry(self._client, "/v1/models", self.model)
        if not m:
            return {}
        ctx = m.get("max_model_len")
        return {"served_context_length": ctx} if ctx is not None else {}
