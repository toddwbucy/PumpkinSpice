"""Decoder-parity gate: greedy decode capture and artifact comparison."""

from __future__ import annotations

import json

import httpx

from pumpkinspice import parity


def _mock_client(handler, base_url: str = "http://x") -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


def test_run_parity_captures_tokens_and_determinism() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v0/models":
            return httpx.Response(200, json={"data": [{"id": "m", "quantization": "Q4_K_M"}]})
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        seen.append(body)
        # greedy decode is deterministic: same prompt -> same tokens
        return httpx.Response(
            200,
            json={"choices": [{"text": " Paris", "logprobs": {"tokens": [" Paris"]}}]},
        )

    client = _mock_client(handler)
    art = parity.run_parity(client, prompts=["The capital of France is"], model="m", max_tokens=8)

    assert art["kind"] == "pumpkinspice.parity"
    assert art["deterministic"] is True
    assert art["fixtures"][0]["tokens"] == [" Paris"]
    # greedy sampler defaults were applied
    assert seen[0]["temperature"] == 0 and seen[0]["top_k"] == 1
    assert seen[0]["max_tokens"] == 8 and seen[0]["model"] == "m"


def test_run_parity_flags_nondeterminism() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        tok = " A" if calls["n"] % 2 else " B"  # differs between the two decodes
        return httpx.Response(200, json={"choices": [{"text": tok, "logprobs": {"tokens": [tok]}}]})

    art = parity.run_parity(_mock_client(handler), prompts=["x"])
    assert art["deterministic"] is False
    assert art["fixtures"][0]["reproducible"] is False


def test_compare_artifacts_pass_and_diverge() -> None:
    a = {"fixtures": [{"prompt": "p", "tokens": ["a", "b", "c"], "text": "abc"}]}
    same = {"fixtures": [{"prompt": "p", "tokens": ["a", "b", "c"], "text": "abc"}]}
    assert parity.compare_artifacts(a, same)["pass"] is True

    diff = {"fixtures": [{"prompt": "p", "tokens": ["a", "X", "c"], "text": "aXc"}]}
    report = parity.compare_artifacts(a, diff)
    assert report["pass"] is False
    assert report["results"][0]["first_divergence"] == 1


def test_compare_artifacts_missing_fixture() -> None:
    a = {"fixtures": [{"prompt": "p", "tokens": ["a"]}]}
    b: dict = {"fixtures": []}
    report = parity.compare_artifacts(a, b)
    assert report["pass"] is False
    assert report["results"][0]["reason"] == "missing in b"
