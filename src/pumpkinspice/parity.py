"""Decoder-parity gate (spec section 4) -- PumpkinSpice's side of the contract.

The same GGUF on two llama.cpp instances is not the same decoder until proven.
This decodes fixed prompts GREEDILY (temperature 0, top-k 1, no repeat penalty,
fixed seed) through LMStudio, records the token stream + environment metadata as
a parity artifact, and self-checks that LMStudio's greedy decode is reproducible.

The artifact is meant to be diffed against the SPU GGUF backend's artifact (run
on the WeaverTools side); ``compare_artifacts`` performs that token-level diff.
This repo implements its side of the shared contract -- it does not invent its
own protocol, and it does not run the SPU.
"""

from __future__ import annotations

from typing import Any

import httpx

from pumpkinspice.model_probe import fetch_model_entry

# Greedy, reproducible decoding. A mismatch against the SPU side is a version or
# sampler skew to pin before any scored run.
GREEDY: dict[str, Any] = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "repeat_penalty": 1.0,
    "seed": 0,
}

DEFAULT_FIXTURES = [
    "The capital of France is",
    "Q: What is 17 + 25? A:",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]


_MODEL_INFO_FIELDS = (
    "id",
    "arch",
    "quantization",
    "state",
    "compatibility_type",
    "max_context_length",
    "loaded_context_length",
)


def _model_info(client: httpx.Client, model: str | None) -> dict[str, Any] | None:
    """Record the loaded model's environment from LMStudio's native endpoint
    (quantization, arch, context). Best-effort -- absent on non-LMStudio servers.
    Shares the query+match with the decoders' capture provenance (model_probe) so the
    two sites cannot drift, and picks the `state`-loaded entry when model is unset."""
    m = fetch_model_entry(client, "/api/v0/models", model)
    if m is None:
        return None
    return {k: m.get(k) for k in _MODEL_INFO_FIELDS}


def _decode(
    client: httpx.Client,
    *,
    prompt: str,
    model: str | None,
    sampler: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens, "logprobs": 1, **sampler}
    if model:
        payload["model"] = model
    resp = client.post("/v1/completions", json=payload)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {"text": choice.get("text", ""), "tokens": list(logprobs.get("tokens") or [])}


def run_parity(
    client: httpx.Client,
    *,
    prompts: list[str],
    model: str | None = None,
    sampler: dict[str, Any] | None = None,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Decode each fixture twice greedily; capture token stream + reproducibility."""
    s = {**GREEDY, **(sampler or {})}
    fixtures: list[dict[str, Any]] = []
    deterministic = True
    for prompt in prompts:
        a = _decode(client, prompt=prompt, model=model, sampler=s, max_tokens=max_tokens)
        b = _decode(client, prompt=prompt, model=model, sampler=s, max_tokens=max_tokens)
        reproducible = a["tokens"] == b["tokens"] if a["tokens"] else a["text"] == b["text"]
        deterministic = deterministic and reproducible
        fixtures.append(
            {
                "prompt": prompt,
                "text": a["text"],
                "tokens": a["tokens"],
                "token_count": len(a["tokens"]),
                "reproducible": reproducible,
            }
        )
    token_level = any(f["token_count"] > 0 for f in fixtures)
    return {
        "kind": "pumpkinspice.parity",
        "side": "lmstudio",
        "metadata": {
            "model": model,
            "sampler": s,
            "max_tokens": max_tokens,
            # Comparison basis: token IDs when LMStudio returns logprobs.tokens,
            # else the decoded text (a strong proxy under greedy decoding).
            "comparison_basis": "tokens" if token_level else "text",
            "lmstudio_model_info": _model_info(client, model),
            # Filled by the apparatus from the two environments:
            "llama_cpp_version": None,
            "spu_llama_cpp_sys_2_version": None,
        },
        "deterministic": deterministic,
        "fixtures": fixtures,
    }


def compare_artifacts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Token-level diff of two parity artifacts (e.g. LMStudio vs SPU). Reports the
    first divergent token per fixture and an overall pass/fail."""
    b_by_prompt = {f["prompt"]: f for f in b.get("fixtures", [])}
    results: list[dict[str, Any]] = []
    overall = True
    for fa in a.get("fixtures", []):
        fb = b_by_prompt.get(fa["prompt"])
        if fb is None:
            results.append({"prompt": fa["prompt"], "match": False, "reason": "missing in b"})
            overall = False
            continue
        ta, tb = fa.get("tokens") or [], fb.get("tokens") or []
        if ta and tb:
            limit = min(len(ta), len(tb))
            divergence = next((i for i in range(limit) if ta[i] != tb[i]), None)
            if divergence is None and len(ta) != len(tb):
                divergence = limit
            match = divergence is None
            results.append(
                {
                    "prompt": fa["prompt"],
                    "match": match,
                    "compared": "tokens",
                    "first_divergence": divergence,
                    "len_a": len(ta),
                    "len_b": len(tb),
                }
            )
        else:
            match = fa.get("text") == fb.get("text")
            results.append({"prompt": fa["prompt"], "match": match, "compared": "text"})
        overall = overall and match
    return {"pass": overall, "results": results}
