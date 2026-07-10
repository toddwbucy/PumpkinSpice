"""Decoder backed by LMStudio's OpenAI-compatible endpoint (spec section 3).

Loads the same GGUF the WeaverTools SPU loads; the decoder-parity gate (spec
section 4) pins sampler + tokenizer settings. This is the pinned-GGUF path used
for scored Benchero 2.0 runs (see the vllm decoder for the baselining path).
Sampler defaults are greedy (temperature 0, top-k 1, no repeat penalty, fixed
seed) so a scored run is reproducible -- pass overrides via the per-call
``sampler`` or config.

Endpoint defaults to the known LAN host (192.168.0.203:1234). Note: that is a
LAN hop, not localhost -- relevant to the transport micro-benchmark (section 5),
which frames the LMStudio path as "localhost".

All the wire logic (HTTP call, sampler merge, usage + chain-of-thought capture)
lives in OpenAICompatDecoder; this subclass only pins the LMStudio-specific
endpoint and the llama.cpp greedy sampler dialect.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pumpkinspice.model_probe import fetch_model_entry
from pumpkinspice.plugins.decoder_openai_compat import OpenAICompatDecoder

DEFAULT_BASE_URL = "http://192.168.0.203:1234"

# Greedy defaults for reproducible / parity-aligned decoding (llama.cpp dialect:
# `top_k: 1` and `repeat_penalty`).
GREEDY: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "repeat_penalty": 1.0,
    "seed": 0,
}


class LMStudioDecoder(OpenAICompatDecoder):
    name = "lmstudio"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    GREEDY: ClassVar[dict[str, Any]] = GREEDY
    # LMStudio decodes whatever model is loaded if `model` is omitted.
    require_model: ClassVar[bool] = False

    def _discover_model_info(self) -> dict[str, Any]:
        """LMStudio's native ``/api/v0/models`` reports the loaded GGUF's quantization, arch,
        and the context length it actually loaded at -- so a pinned-GGUF scored run records
        them without the operator declaring them (and a silent context downgrade is caught).
        When no model is configured (LMStudio's decode-whatever-is-loaded default) the shared
        probe selects the entry whose ``state`` is loaded, not just the first of every
        downloaded model; ``state`` is recorded for audit. Best-effort ``{}`` on any error;
        server-verified fields win over any declared ``quantization`` since they reflect what
        is really loaded."""
        m = fetch_model_entry(self._client, "/api/v0/models", self.model)
        if not m:
            return {}
        field_map = {
            "quantization": "quantization",
            "arch": "arch",
            "state": "state",
            "loaded_context_length": "served_context_length",
        }
        return {dst: m[src] for src, dst in field_map.items() if m.get(src) is not None}
