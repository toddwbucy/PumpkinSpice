"""pgvector retrieval: scoped-DSN enforcement, fully mocked query paths, and the
embed-provenance gate (findings #1-#10).

Skipped unless the pgvector extra is installed (CI installs it). No live Postgres:
construction does zero DB IO (the provenance check is lazy, on first retrieve), and
the query paths run against fakes."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from pumpkinspice.plugins.retrieval_pgvector import PgVectorRetrieval


def _fake_psycopg(
    main_rows: list[tuple],
    *,
    stamps: list[str] | None = ("nomic-embed-text",),
    dim: int | None = 3,
    recipe_rows: list[tuple] | None = None,
    capture: dict[str, object] | None = None,
) -> types.SimpleNamespace:
    """A psycopg stand-in whose cursor routes by SQL text: the DISTINCT-stamp and
    vector_dims provenance reads, an optional recipe-book query, else the main query.
    ``connect`` accepts the connect_timeout kwarg the plugin now passes; the main
    (``::vector``) query's sql+params land in ``capture`` for regression assertions."""

    class FakeCursor:
        last = ""

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def execute(self, sql: object, params: object = None) -> None:
            self.last = str(sql)
            if capture is not None and "::vector" in self.last:
                capture["sql"], capture["params"] = sql, params

        def fetchall(self) -> list[tuple]:
            if "DISTINCT" in self.last:
                return [(s,) for s in (stamps or [])]
            if "recipe_book" in self.last and recipe_rows is not None:
                return recipe_rows
            return main_rows

        def fetchone(self) -> tuple | None:
            if "vector_dims" in self.last:
                return None if dim is None else (dim,)
            return None

    class FakeConn:
        def __enter__(self) -> FakeConn:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    return types.SimpleNamespace(connect=lambda dsn, **kw: FakeConn())


def _plugin(monkeypatch: pytest.MonkeyPatch, fake: object, **cfg: object) -> PgVectorRetrieval:
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    base = {"dsn_env": "PUMPKINSPICE_PG_DSN", "table": "kg.belief_nodes"}
    p = PgVectorRetrieval({**base, **cfg})
    monkeypatch.setattr(p, "_embed", lambda q: [0.1, 0.2, 0.3])  # 3-dim query
    p._psycopg = fake  # type: ignore[assignment]
    p._register_vector = lambda conn: None  # type: ignore[assignment,method-assign]
    return p


def test_requires_scoped_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUMPKINSPICE_PG_DSN", raising=False)
    with pytest.raises(RuntimeError, match="scoped read-only DSN"):
        PgVectorRetrieval({"dsn_env": "PUMPKINSPICE_PG_DSN"})


def test_rejects_malicious_identifier_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config-supplied identifiers must be plain SQL identifiers: an injection
    attempt fails loudly at construction, before any query is composed."""
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    for bad in (
        {"table": "belief_nodes; DROP TABLE x--"},
        {"vector_column": 'embedding" <=> NULL; --'},
        {"relational_schema": "kg; DROP SCHEMA kg--"},
        {"id_column": "kg.id"},  # only `table` may be schema-qualified
        {"table": "kg.belief_nodes; --"},  # the part after the dot is checked too
    ):
        with pytest.raises(ValueError, match="not a valid SQL identifier"):
            PgVectorRetrieval(dict(bad))


def test_rejects_bad_check_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    with pytest.raises(ValueError, match="embed_model_check must be one of"):
        PgVectorRetrieval({"dsn_env": "PUMPKINSPICE_PG_DSN", "embed_model_check": "loud"})


def test_construction_does_no_db_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provenance check is lazy, so a valid construction opens no DB connection --
    unit tests need no live Postgres (CLAUDE.md's hermetic-tests contract)."""
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    import psycopg

    calls: list[object] = []
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: calls.append(a))
    PgVectorRetrieval({"dsn_env": "PUMPKINSPICE_PG_DSN", "table": "kg.belief_nodes"})
    assert calls == []  # no connect() at construction


def test_retrieve_maps_rows_to_belief_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fake = _fake_psycopg([("n1", "hello", {}, 0.9), ("n2", "world", {}, 0.8)], capture=captured)
    p = _plugin(monkeypatch, fake)

    res = p.retrieve("q", top_k=2)
    assert [n.id for n in res.nodes] == ["n1", "n2"]
    assert res.nodes[0].score == 0.9
    assert res.backend == "pgvector"

    from psycopg.sql import Composed

    query = captured["sql"]
    assert isinstance(query, Composed)
    assert query.as_string() == (
        'SELECT "id", "text", "metadata", 1 - ("embedding" <=> %s::vector) AS score '
        'FROM "kg"."belief_nodes" ORDER BY "embedding" <=> %s::vector LIMIT %s'
    )
    params = captured["params"]
    assert isinstance(params, tuple)
    assert params[0] == "[0.1,0.2,0.3]" and params[2] == 2


def test_provenance_model_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # corpus stamped with a different model than configured -> fail fast at first retrieve
    fake = _fake_psycopg([("n1", "x", {}, 0.9)], stamps=["some-other-model"])
    p = _plugin(monkeypatch, fake)
    with pytest.raises(ValueError, match="embed-model mismatch"):
        p.retrieve("q", top_k=1)


def test_provenance_dimension_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # stored vectors are 768-dim, the query embedder yields 3-dim -> definitive mismatch
    fake = _fake_psycopg([("n1", "x", {}, 0.9)], stamps=["nomic-embed-text"], dim=768)
    p = _plugin(monkeypatch, fake)
    with pytest.raises(ValueError, match="DIMENSION mismatch"):
        p.retrieve("q", top_k=1)


def test_provenance_mixed_corpus_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # a partial re-seed left two distinct stamps -> LIMIT-1 would miss it; DISTINCT catches it
    fake = _fake_psycopg([("n1", "x", {}, 0.9)], stamps=["nomic-embed-text", "old-model"])
    p = _plugin(monkeypatch, fake)
    with pytest.raises(ValueError, match="MIXED embed models"):
        p.retrieve("q", top_k=1)


def test_provenance_off_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # escape hatch: a known-same-space rename runs despite the name mismatch
    fake = _fake_psycopg([("n1", "x", {}, 0.9)], stamps=["some-other-model"])
    p = _plugin(monkeypatch, fake, embed_model_check="off")
    res = p.retrieve("q", top_k=1)
    assert [n.id for n in res.nodes] == ["n1"]


def test_provenance_unstamped_corpus_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # a legacy (pre-provenance) corpus has no stamps and matching dims -> no raise
    fake = _fake_psycopg([("n1", "x", {}, 0.9)], stamps=[], dim=3)
    p = _plugin(monkeypatch, fake)
    res = p.retrieve("q", top_k=1)
    assert [n.id for n in res.nodes] == ["n1"]


def test_relational_mode_appends_recipe_book(monkeypatch: pytest.MonkeyPatch) -> None:
    RECIPE = (
        "Recipe for copper_dagger (Copper Dagger, level 1 weapon):\n"
        "Skills required: mining level 1, weaponcrafting level 1.\n"
        "Ordered steps (do gathers first, then craft bottom-up):\n"
        "  1. Gather 48x copper_ore at copper_rocks (2,0).\n"
        "  2. Craft 6x copper at the mining workshop (1,5) (requires mining level 1).\n"
        "  3. Craft 1x copper_dagger at the weaponcrafting workshop (2,1) "
        "(requires weaponcrafting level 1)."
    )
    main = [
        ("item:copper_dagger", "Copper Dagger", {"kind": "item", "code": "copper_dagger"}, 0.91)
    ]
    fake = _fake_psycopg(main, recipe_rows=[("copper_dagger", RECIPE)])
    p = _plugin(monkeypatch, fake, relational=True)

    res = p.retrieve("how do I craft a copper dagger", top_k=5)
    assert res.backend == "pgvector+relational"
    texts = [n.text for n in res.nodes]
    assert any("Copper Dagger" in t for t in texts)
    assert any("Gather 48x copper_ore at copper_rocks (2,0)" in t for t in texts)
    assert any("requires mining level 1" in t for t in texts)
    relations = {n.metadata.get("relation") for n in res.nodes if n.metadata.get("relation")}
    assert relations == {"recipe_book"}
