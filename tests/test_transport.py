"""Transport micro-benchmark: distribution summary and the mocked run loop."""

from __future__ import annotations

import httpx

from pumpkinspice import transport


def test_summarize_percentiles() -> None:
    s = transport.summarize([float(i) for i in range(1, 101)])  # 1..100 ms
    assert s["count"] == 100
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 100.0
    assert s["p50_ms"] == 50.5  # interpolated median of 1..100
    assert abs(s["mean_ms"] - 50.5) < 1e-9
    assert s["p90_ms"] > s["p50_ms"] < s["p99_ms"]


def test_summarize_empty() -> None:
    assert transport.summarize([]) == {"count": 0}


def test_run_transport_structure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    art = transport.run_transport(client, iterations=5, warmup=2, model="m")

    assert art["kind"] == "pumpkinspice.transport"
    assert art["metadata"]["iterations"] == 5
    assert art["minimal_decode_ms"]["count"] == 5
    assert art["models_ping_ms"]["count"] == 5
