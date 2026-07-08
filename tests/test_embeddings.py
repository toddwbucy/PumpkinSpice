"""Tests for the shared embeddings helper (used by both retrieval arms + seeders)."""

from __future__ import annotations

import httpx

from pumpkinspice.embeddings import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    embed_query,
    warm_up,
)


def _client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")


def test_defaults_point_at_ollama() -> None:
    assert DEFAULT_EMBED_URL == "http://localhost:11434"
    assert DEFAULT_EMBED_MODEL == "nomic-embed-text"


def test_embed_query_sends_model_and_parses_vector() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert request.url.path == "/v1/embeddings"
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    vec = embed_query(_client(handler), "nomic-embed-text", "copper dagger")
    assert vec == [0.1, 0.2, 0.3]
    assert seen == {"input": "copper dagger", "model": "nomic-embed-text"}


def test_embed_query_omits_model_when_none() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    embed_query(_client(handler), None, "q")
    assert "model" not in seen  # lets a loaded-model server pick its own


def test_warm_up_swallows_a_down_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    warm_up(_client(handler), "nomic-embed-text")  # must not raise
