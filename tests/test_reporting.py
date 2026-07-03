"""The SQLite run registry behind the Reports tab."""

from __future__ import annotations

from pathlib import Path

from pumpkinspice.reporting import RunRegistry


def _run(rid: str, model: str, steps: int, success: bool, **kw: object) -> dict:
    return {
        "id": rid,
        "benchmark": "herobench",
        "model": model,
        "strategy": kw.get("strategy", "reactive"),
        "retrieval": "pgvector+relational",
        "task": "Craft a copper dagger.",
        "goal": "copper_dagger",
        "max_turns": 50,
        "started_at": "2026-06-28T00:00:00",
        "finished_at": kw.get("finished_at", "2026-06-28T00:10:00"),
        "status": "done",
        "metrics": {"steps": steps, "success": success, "failed_actions": 0},
        "capture_path": f"captures/{rid}.jsonl",
        "tags": kw.get("tags", []),
    }


def test_record_list_get(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "results.db")
    reg.record(_run("r1", "mistral-24b", 15, True))
    reg.record(_run("r2", "ministral-14b", 14, True, tags=["baseline"]))

    runs = reg.list_runs(benchmark="herobench")
    assert {r["id"] for r in runs} == {"r1", "r2"}
    # metrics + tags parse back out of JSON
    r2 = reg.get("r2")
    assert r2 is not None and r2["metrics"]["steps"] == 14 and r2["tags"] == ["baseline"]
    assert reg.get("nope") is None
    # filter by model and by tag
    assert [r["id"] for r in reg.list_runs(model="mistral-24b")] == ["r1"]
    assert [r["id"] for r in reg.list_runs(tag="baseline")] == ["r2"]


def test_record_is_upsert_and_persists(tmp_path: Path) -> None:
    db = tmp_path / "results.db"
    RunRegistry(db).record(_run("r1", "m", 20, False))
    # re-open (survives "restart") and overwrite the same id
    reg = RunRegistry(db)
    reg.record(_run("r1", "m", 12, True))
    runs = reg.list_runs()
    assert len(runs) == 1 and runs[0]["metrics"]["steps"] == 12


def test_update_tags_label_notes(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "results.db")
    reg.record(_run("r1", "m", 15, True))
    assert reg.update("r1", label="best run", tags=["s3", "keep"], notes="clean") is True
    r = reg.get("r1")
    assert r is not None
    assert r["label"] == "best run" and r["tags"] == ["s3", "keep"] and r["notes"] == "clean"
    assert reg.update("missing", label="x") is False  # unknown id


def test_leaderboard(tmp_path: Path) -> None:
    reg = RunRegistry(tmp_path / "results.db")
    reg.record(_run("a1", "ministral-14b", 14, True))
    reg.record(_run("b1", "mistral-24b", 15, True))
    reg.record(_run("b2", "mistral-24b", 17, False))  # a failed run drags success rate
    board = reg.leaderboard(benchmark="herobench")
    # ranked by best steps -> ministral first
    assert board[0]["model"] == "ministral-14b" and board[0]["best_steps"] == 14
    mistral = next(b for b in board if b["model"] == "mistral-24b")
    assert mistral["runs"] == 2 and mistral["successes"] == 1 and mistral["success_rate"] == 0.5
    assert mistral["best_steps"] == 15  # only successful runs count toward steps
