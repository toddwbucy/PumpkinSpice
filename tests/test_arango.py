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


def _provenance_client(stamps: list[object], dim: object):  # type: ignore[no-untyped-def]
    """A fake ArangoClient whose AQL routes the DISTINCT-stamp and LENGTH-dim provenance
    reads to fixed values, and every other query to one node row."""

    class FakeAQL:
        def execute(self, aql: str, bind_vars: dict[str, object]) -> object:
            if "DISTINCT" in aql:
                assert bind_vars["key"] == "embed_model"  # key is BOUND, not hardcoded
                return iter(stamps)
            if "LENGTH(doc.embedding)" in aql:  # the dim probe (the main AQL also has LENGTH(@q))
                return iter([dim])
            return iter([{"id": "n1", "text": "x", "score": 0.9}])

    class FakeDB:
        aql = FakeAQL()

    class FakeClient:
        def __init__(self, hosts: str) -> None:
            pass

        def db(self, database: str, username: str, password: str) -> FakeDB:
            return FakeDB()

    return FakeClient


def _arango(monkeypatch: pytest.MonkeyPatch, client: object, **cfg: object) -> object:
    monkeypatch.setenv("ARANGO_AGENT_USER", "ps_agent_ro")
    monkeypatch.setenv("ARANGO_AGENT_PASSWORD", "secret")
    monkeypatch.setattr(arango, "ArangoClient", client)
    p = ArangoRetrieval({"collection": "belief_nodes", **cfg})
    monkeypatch.setattr(p, "_embed", lambda q: [0.1, 0.2, 0.3])  # 3-dim query
    return p


def test_provenance_model_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _arango(monkeypatch, _provenance_client(["some-other-model"], 3))
    with pytest.raises(ValueError, match="embed-model mismatch"):
        p.retrieve("q", top_k=1)


def test_provenance_dimension_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # the arango arm has no AQL dimension guard, so this is the check that protects it
    p = _arango(monkeypatch, _provenance_client(["nomic-embed-text"], 768))
    with pytest.raises(ValueError, match="DIMENSION mismatch"):
        p.retrieve("q", top_k=1)


def test_provenance_off_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _arango(monkeypatch, _provenance_client(["some-other-model"], 3), embed_model_check="off")
    res = p.retrieve("q", top_k=1)
    assert [n.id for n in res.nodes] == ["n1"]
