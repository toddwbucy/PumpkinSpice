"""Transport micro-benchmark (spec section 5) -- PumpkinSpice's side.

Measures decoder round-trip latency over the LMStudio endpoint on a fixed prompt
set, to bound and rule out transport as a differentiator (expected negligible).
Reports the full distribution (percentiles), not just the mean.

Two measures: a pure-transport ``models_ping`` (GET /v1/models, no generation)
and a ``minimal_decode`` (max_tokens=1) round-trip. The WeaverTools unix-socket
path is measured on the SPU side; this is the HTTP side for comparison.

Note: the LMStudio host may be a LAN hop (192.168.0.203), not loopback -- the
artifact records the endpoint so a LAN-vs-localhost difference is visible.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import httpx


def _percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_samples[int(k)]
    return sorted_samples[lo] * (hi - k) + sorted_samples[hi] * (k - lo)


def summarize(samples: list[float]) -> dict[str, float]:
    """Distribution summary in milliseconds (not just the mean -- spec section 5)."""
    if not samples:
        return {"count": 0}
    s = sorted(samples)
    return {
        "count": len(s),
        "min_ms": s[0],
        "p50_ms": _percentile(s, 0.50),
        "p90_ms": _percentile(s, 0.90),
        "p99_ms": _percentile(s, 0.99),
        "max_ms": s[-1],
        "mean_ms": sum(s) / len(s),
        "stdev_ms": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


def run_transport(
    client: httpx.Client,
    *,
    prompt: str = "ping",
    iterations: int = 50,
    warmup: int = 5,
    max_tokens: int = 1,
    model: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": 0,
    }
    if model:
        payload["model"] = model

    for _ in range(warmup):
        client.post("/v1/chat/completions", json=payload).raise_for_status()

    decode: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        client.post("/v1/chat/completions", json=payload).raise_for_status()
        decode.append((time.perf_counter() - t0) * 1e3)

    ping: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        client.get("/v1/models").raise_for_status()
        ping.append((time.perf_counter() - t0) * 1e3)

    return {
        "kind": "pumpkinspice.transport",
        "metadata": {
            "endpoint": str(client.base_url),
            "model": model,
            "iterations": iterations,
            "warmup": warmup,
            "max_tokens": max_tokens,
        },
        "models_ping_ms": summarize(ping),
        "minimal_decode_ms": summarize(decode),
    }
