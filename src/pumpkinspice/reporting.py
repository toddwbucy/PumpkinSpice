"""Persistent run registry for the Reports tab (web PRD section 7).

A finished run's config + computed metrics + operator tags/label/notes are written to
a SQLite file (``captures/results.db``), separate from the KG. Raw per-turn data stays
in the capture JSONL referenced by ``capture_path``; the registry is the queryable
index the Reports tab reads (so it does not recompute metrics on every view).

Pure SQLite over stdlib ``sqlite3`` (no extra dependency); a connection is opened per
call, which is fine for the low write volume and keeps it thread-safe under the web
RunManager. ``metrics`` and ``tags`` are stored as JSON text and parsed back out.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_COLUMNS = [
    "id",
    "benchmark",
    "model",
    "strategy",
    "retrieval",
    "task",
    "goal",
    "max_turns",
    "started_at",
    "finished_at",
    "status",
    "metrics",
    "capture_path",
    "label",
    "tags",
    "notes",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    benchmark    TEXT,
    model        TEXT,
    strategy     TEXT,
    retrieval    TEXT,
    task         TEXT,
    goal         TEXT,
    max_turns    INTEGER,
    started_at   TEXT,
    finished_at  TEXT,
    status       TEXT,
    metrics      TEXT DEFAULT '{}',
    capture_path TEXT,
    label        TEXT DEFAULT '',
    tags         TEXT DEFAULT '[]',
    notes        TEXT DEFAULT ''
)
"""

_SORTABLE = {
    "finished_at",
    "started_at",
    "model",
    "strategy",
    "status",
}


class RunRegistry:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _from_row(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metrics"] = json.loads(d.get("metrics") or "{}")
        d["tags"] = json.loads(d.get("tags") or "[]")
        return d

    def record(self, run: dict[str, Any]) -> None:
        """Insert or replace a run record. ``metrics`` (a dict) and ``tags`` (a list)
        are JSON-serialized; unknown keys are ignored, missing ones default."""
        data = dict(run)
        data["metrics"] = json.dumps(data.get("metrics") or {})
        data["tags"] = json.dumps(data.get("tags") or [])
        values = [data.get(col) for col in _COLUMNS]
        placeholders = ",".join(["?"] * len(_COLUMNS))
        with self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO runs ({','.join(_COLUMNS)}) VALUES ({placeholders})",
                values,
            )

    def list_runs(
        self,
        *,
        benchmark: str | None = None,
        model: str | None = None,
        strategy: str | None = None,
        tag: str | None = None,
        sort: str = "finished_at",
        desc: bool = True,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        for col, val in (("benchmark", benchmark), ("model", model), ("strategy", strategy)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        order = sort if sort in _SORTABLE else "finished_at"
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order} {'DESC' if desc else 'ASC'}"
        with self._conn() as c:
            rows = [self._from_row(r) for r in c.execute(sql, params)]
        if tag is not None:
            rows = [r for r in rows if tag in r["tags"]]
        return rows

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._from_row(row) if row is not None else None

    def update(
        self,
        run_id: str,
        *,
        label: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> bool:
        sets: list[str] = []
        params: list[Any] = []
        if label is not None:
            sets.append("label = ?")
            params.append(label)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if notes is not None:
            sets.append("notes = ?")
            params.append(notes)
        if not sets:
            return False
        params.append(run_id)
        with self._conn() as c:
            cur = c.execute(f"UPDATE runs SET {','.join(sets)} WHERE id = ?", params)
            return cur.rowcount > 0

    def leaderboard(self, *, benchmark: str | None = None) -> list[dict[str, Any]]:
        """Per-model aggregate: run count, success rate, best and average
        steps-to-completion (over successful runs). Ranked by best steps."""
        by_model: dict[str, list[dict[str, Any]]] = {}
        for r in self.list_runs(benchmark=benchmark):
            by_model.setdefault(r["model"] or "?", []).append(r)
        out: list[dict[str, Any]] = []
        for model, runs in by_model.items():
            succ = [r for r in runs if (r["metrics"] or {}).get("success") is True]
            steps = [
                int(r["metrics"]["steps"])
                for r in succ
                if isinstance(r["metrics"].get("steps"), int)
            ]
            out.append(
                {
                    "model": model,
                    "runs": len(runs),
                    "successes": len(succ),
                    "success_rate": (len(succ) / len(runs)) if runs else 0.0,
                    "best_steps": min(steps) if steps else None,
                    "avg_steps": (sum(steps) / len(steps)) if steps else None,
                }
            )
        out.sort(
            key=lambda x: (x["best_steps"] is None, x["best_steps"] if x["best_steps"] else 1e9)
        )
        return out
