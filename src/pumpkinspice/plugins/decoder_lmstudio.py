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
