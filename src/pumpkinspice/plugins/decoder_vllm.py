"""Decoder backed by a vLLM OpenAI-compatible server (`vllm serve`).

This is the *baselining / floor-test* decoder path, NOT the scored-run path.
Scored Benchero 2.0 runs stay on the pinned-GGUF LMStudio decoder so the
decoder-parity contract with the WeaverTools SPU (llama.cpp) is untouched; vLLM
is adopted purely on serving merits (honest --max-model-len so there is no silent
context truncation, one resident model with no JIT-evict thrash, a true localhost
path). It exposes no per-layer hidden states -- the #7/#8 trajectory-geometry
metrics are computed offline by a separate transformers replay rig, not here.

Two things differ from LMStudio and are why this is its own subclass:

* ``model`` is REQUIRED. vLLM's /v1/chat/completions 400s without a `model` that
  matches the served name; there is no loaded-model fallback. Enforced in the base
  class via ``require_model``.
* Sampler dialect. vLLM spells the greedy knobs ``repetition_penalty`` (not
  llama.cpp's ``repeat_penalty``) and treats ``top_k: -1`` as "consider all"
  (llama.cpp uses ``1``). vLLM accepts these as top-level request-body fields.

Defaults to 127.0.0.1:8001 so it does not collide with the HeroBench world server
(127.0.0.1:8000).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pumpkinspice.plugins.decoder_openai_compat import OpenAICompatDecoder

# 8001, not vLLM's own default 8000, to avoid colliding with HeroBench's world API.
DEFAULT_BASE_URL = "http://127.0.0.1:8001"

# Greedy defaults in vLLM's dialect. temperature 0 already forces greedy; the rest
# are set explicitly so a run is reproducible and matches the LMStudio greedy
# intent (top_k -1 = all, repetition_penalty 1.0 = off, fixed seed).
GREEDY: dict[str, Any] = {
    "temperature": 0,
    "top_k": -1,
    "top_p": 1,
    "repetition_penalty": 1.0,
    "seed": 0,
}


class VLLMDecoder(OpenAICompatDecoder):
    name = "vllm"
    DEFAULT_BASE_URL: ClassVar[str] = DEFAULT_BASE_URL
    GREEDY: ClassVar[dict[str, Any]] = GREEDY
    require_model: ClassVar[bool] = True
