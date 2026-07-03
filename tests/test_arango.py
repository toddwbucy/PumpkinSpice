"""arango retrieval: scoped-credential enforcement and a mocked query path.

Skipped unless the arango extra is installed (CI installs it)."""

from __future__ import annotations

import pytest

arango = pytest.importorskip("arango")

from pumpkinspice.plugins.retrieval_arango import ArangoRetrieval  # noqa: E402


def test_requires_scoped_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARANGO_AGENT_USER", raising=False)
    monkeypatch.delenv("ARANGO_AGENT_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="scoped read-only user"):
        ArangoRetrieval({})


def test_rejects_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARANGO_AGENT_USER", "root")
    monkeypatch.setenv("ARANGO_AGENT_PASSWORD", "x")
    with pytest.raises(RuntimeError, match="must not run as root"):
        ArangoRetrieval({})


def test_retrieve_maps_rows_and_binds_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARANGO_AGENT_USER", "ps_agent_ro")
    monkeypatch.setenv("ARANGO_AGENT_PASSWORD", "secret")
    captured: dict[str, object] = {}

    class FakeAQL:
        def execute(self, aql: str, bind_vars: dict[str, object]) -> object:
            captured["aql"] = aql
            captured["bind"] = bind_vars
            return iter(
                [
                    {"id": "n1", "text": "hello", "score": 0.9},
                    {"id": "n2", "text": "world", "score": 0.8},
                ]
            )

    class FakeDB:
        aql = FakeAQL()

    class FakeClient:
        def __init__(self, hosts: str) -> None:
            pass

        def db(self, database: str, username: str, password: str) -> FakeDB:
            return FakeDB()

    monkeypatch.setattr(arango, "ArangoClient", FakeClient)

    p = ArangoRetrieval({"collection": "belief_nodes"})
    monkeypatch.setattr(p, "_embed", lambda q: [0.1, 0.2, 0.3])

    res = p.retrieve("q", top_k=2)
    assert [n.id for n in res.nodes] == ["n1", "n2"]
    assert res.nodes[0].score == 0.9
    assert res.backend == "arango"

    bind = captured["bind"]
    assert isinstance(bind, dict)
    assert bind["@collection"] == "belief_nodes"
    assert bind["k"] == 2
    assert "APPROX_NEAR" not in str(captured["aql"])  # plain cosine, no native vector index
    # Cross-backend id parity: the AQL must surface the ORIGINAL node id field
    # (e.g. "mechanic:crafting_xp"), not the sanitized _key, so arango and
    # pgvector captures identify the same belief node the same way.
    assert "doc.id != null ? doc.id : doc._key" in str(captured["aql"])
