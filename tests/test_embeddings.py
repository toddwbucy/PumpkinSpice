"""Tests for the shared embeddings helper (used by both retrieval arms + seeders)."""

from __future__ import annotations

import httpx
import pytest

from pumpkinspice.embeddings import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    check_embed_provenance,
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


def _fetch(stamps: list[str], dim: int | None = 3):  # type: ignore[no-untyped-def]
    return lambda: (stamps, dim)


def test_provenance_passes_on_match_and_unstamped() -> None:
    check_embed_provenance(_fetch(["nomic-embed-text"]), "nomic-embed-text", 3)  # match
    check_embed_provenance(_fetch([]), "nomic-embed-text", 3)  # unstamped legacy corpus
    check_embed_provenance(_fetch(["x"]), None, 3)  # falsy configured = server-picks -> skip


def test_provenance_raises_on_name_mismatch() -> None:
    with pytest.raises(ValueError, match="embed-model mismatch"):
        check_embed_provenance(_fetch(["nomic-embed-text"]), "text-embedding-nomic-v1.5", 3)


def test_provenance_raises_on_mixed_corpus() -> None:
    with pytest.raises(ValueError, match="MIXED embed models"):
        check_embed_provenance(_fetch(["nomic-embed-text", "old-model"]), "nomic-embed-text", 3)


def test_provenance_raises_on_dimension_mismatch() -> None:
    # dimension is checked first and is name-independent
    with pytest.raises(ValueError, match="DIMENSION mismatch"):
        check_embed_provenance(_fetch(["nomic-embed-text"], dim=768), "nomic-embed-text", 3)


def test_provenance_modes_warn_and_off(caplog) -> None:  # type: ignore[no-untyped-def]
    import logging

    # warn: logs, does not raise
    with caplog.at_level(logging.WARNING):
        check_embed_provenance(_fetch(["other"]), "nomic-embed-text", 3, mode="warn")
    assert any("mismatch" in r.message for r in caplog.records)
    # off: silent, does not raise
    check_embed_provenance(_fetch(["other"]), "nomic-embed-text", 3, mode="off")


def test_provenance_skips_and_logs_when_fetch_raises() -> None:
    def boom() -> tuple[list[str], int | None]:
        raise RuntimeError("db down")

    check_embed_provenance(boom, "nomic-embed-text", 3)  # swallowed + logged, not fatal
