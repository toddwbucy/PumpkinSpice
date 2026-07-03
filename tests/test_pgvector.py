"""pgvector retrieval: scoped-DSN enforcement and fully mocked query paths.

Skipped unless the pgvector extra is installed (CI installs it)."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from pumpkinspice.plugins.retrieval_pgvector import PgVectorRetrieval


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


def test_retrieve_maps_rows_to_belief_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    p = PgVectorRetrieval({"dsn_env": "PUMPKINSPICE_PG_DSN", "table": "kg.belief_nodes"})

    monkeypatch.setattr(p, "_embed", lambda q: [0.1, 0.2, 0.3])
    captured: dict[str, object] = {}

    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def execute(self, sql: object, params: object) -> None:
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self) -> list[tuple[str, str, dict, float]]:
            # rows are (id, text, metadata, score)
            return [("n1", "hello", {}, 0.9), ("n2", "world", {}, 0.8)]

    class FakeConn:
        def __enter__(self) -> FakeConn:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    p._psycopg = types.SimpleNamespace(connect=lambda dsn: FakeConn())
    p._register_vector = lambda conn: None

    res = p.retrieve("q", top_k=2)
    assert [n.id for n in res.nodes] == ["n1", "n2"]
    assert res.nodes[0].score == 0.9
    assert res.backend == "pgvector"
    assert res.latency_ms >= 0.0

    # Regression guard: the query vector must be cast to ::vector and passed as a
    # text literal -- a bare list is sent as double precision[], which has no
    # `<=>` operator against a vector column.
    assert "::vector" in str(captured["sql"])
    # Identifiers are psycopg sql.Identifier-composed (injection-safe) and must
    # render to the same query the old f-string produced (modulo quoting).
    from psycopg.sql import Composed

    query = captured["sql"]
    assert isinstance(query, Composed)
    assert query.as_string() == (
        'SELECT "id", "text", "metadata", 1 - ("embedding" <=> %s::vector) AS score '
        'FROM "kg"."belief_nodes" ORDER BY "embedding" <=> %s::vector LIMIT %s'
    )
    params = captured["params"]
    assert isinstance(params, tuple)
    assert params[0] == "[0.1,0.2,0.3]"
    assert params[2] == 2  # top_k


def test_relational_mode_appends_recipe_book(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUMPKINSPICE_PG_DSN", "postgresql://u:p@localhost/db")
    p = PgVectorRetrieval({"dsn_env": "PUMPKINSPICE_PG_DSN", "relational": True})
    monkeypatch.setattr(p, "_embed", lambda q: [0.1, 0.2, 0.3])

    # The precomputed recipe-book entry: the full flattened chain (the runtime just
    # looks it up; the graph walk happened build-side in seed_recipe_book.py).
    RECIPE = (
        "Recipe for copper_dagger (Copper Dagger, level 1 weapon):\n"
        "Skills required: mining level 1, weaponcrafting level 1.\n"
        "Ordered steps (do gathers first, then craft bottom-up):\n"
        "  1. Gather 48x copper_ore at copper_rocks (2,0).\n"
        "  2. Craft 6x copper at the mining workshop (1,5) (requires mining level 1).\n"
        "  3. Craft 1x copper_dagger at the weaponcrafting workshop (2,1) "
        "(requires weaponcrafting level 1)."
    )

    class FakeCursor:
        last = ""

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def execute(self, sql: object, params: object = None) -> None:
            self.last = str(sql)  # composed queries stringify with their parts visible

        def fetchall(self) -> list[tuple]:
            if "recipe_book" in self.last:
                return [("copper_dagger", RECIPE)]
            # the vector query: one item-kind hit
            return [
                (
                    "item:copper_dagger",
                    "Copper Dagger",
                    {"kind": "item", "code": "copper_dagger"},
                    0.91,
                )
            ]

    class FakeConn:
        def __enter__(self) -> FakeConn:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def cursor(self) -> FakeCursor:
            return FakeCursor()

    p._psycopg = types.SimpleNamespace(connect=lambda dsn: FakeConn())
    p._register_vector = lambda conn: None

    res = p.retrieve("how do I craft a copper dagger", top_k=5)
    assert res.backend == "pgvector+relational"
    texts = [n.text for n in res.nodes]
    # the semantic hit plus its full recipe-book chain (multi-hop, with levels)
    assert any("Copper Dagger" in t for t in texts)
    assert any("Gather 48x copper_ore at copper_rocks (2,0)" in t for t in texts)
    assert any("requires mining level 1" in t for t in texts)  # smelt step + level
    relations = {n.metadata.get("relation") for n in res.nodes if n.metadata.get("relation")}
    assert relations == {"recipe_book"}
