"""Run manager: launch a HeroBench run in a background thread and buffer its
per-turn records so the API can stream them live (SSE) and serve them after."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx

from .. import analyze, kernel
from ..config import RunConfig
from ..contracts import Turn
from ..loop import AgentLoop
from ..reporting import RunRegistry


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def reset_herobench_character(base_url: str, character: str) -> None:
    """Reset a HeroBench character to a fresh L1 / empty-inventory baseline (pure
    REST: delete then create). Verified 2026-06-29 that the REST delete clears the
    inventory hash too, so no redis access is needed. Used between trials so each
    starts from the identical line."""
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=20.0) as c:
        # delete takes the name as a raw JSON-string body; 498 if it did not exist.
        with contextlib.suppress(httpx.HTTPError):
            c.post("/characters/delete", json=character)
        c.post("/characters/create", json={"name": character, "skin": "men2"})


def _stochastic_sampler(temperature: float, seed: int) -> dict[str, Any]:
    """A genuinely stochastic sampler for trials. The decoder's GREEDY default pins
    top_k=1 (which collapses to one token and makes temperature inert) and seed=0;
    override both so trials actually diverge yet stay per-seed reproducible."""
    return {"temperature": temperature, "top_k": 0, "top_p": 0.95, "seed": seed}


@dataclasses.dataclass
class RunRecord:
    id: str
    config_name: str
    task: str
    plugins: dict[str, str]
    status: str = "running"  # running | done | error
    turns: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    error: str | None = None
    # Reporting metadata (persisted to the run registry when the run finishes).
    benchmark: str = "herobench"
    model: str = ""
    strategy: str = ""
    retrieval: str = ""
    goal: str = ""
    max_turns: int = 0
    capture_path: str = ""
    started_at: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)  # e.g. batch:<id>, seed:<i>


class _WebCapture:
    """Capture sink that buffers turns into a RunRecord (and mirrors to JSONL)."""

    name = "web"

    def __init__(self, record: RunRecord, jsonl_path: Path | None) -> None:
        self._record = record
        self._fh = None
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = jsonl_path.open("w", encoding="utf-8")

    def record(self, turn: Turn) -> None:
        import json

        row = dataclasses.asdict(turn)
        self._record.turns.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()


def _build_loop(cfg: RunConfig, capture: _WebCapture) -> AgentLoop:
    def plug(slot: str) -> Any:
        return kernel.load_plugin(slot, cfg.plugin_name(slot), cfg.slot_config(slot))

    return AgentLoop(
        decoder=plug("decoder"),
        retrieval=plug("retrieval"),
        world=plug("world"),
        prompt=plug("prompt"),
        capture=capture,
        task=cfg.task,
        top_k=int(cfg.slot_config("retrieval").get("top_k", 5)),
        sampler=cfg.slot_config("decoder").get("sampler", {}),
        history_window=int(cfg.run.get("history_window", 0)),
        goal_item=cfg.run.get("goal_item"),
        goal_level=cfg.run.get("goal_level"),
        goal_skill=cfg.run.get("goal_skill"),
        goal_state_key=cfg.run.get("goal_state_key"),
        goal_monster=cfg.run.get("goal_monster"),
    )


class RunManager:
    def __init__(self, captures_dir: Path, registry: RunRegistry | None = None) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._stops: dict[str, threading.Event] = {}  # run_id -> cooperative stop flag
        self._batch_stops: dict[str, threading.Event] = {}  # batch_id -> abort flag
        # Guards the three dicts above, which are mutated from API handler
        # threads and background run/batch threads. RLock because stop_batch()
        # calls stop() while holding it. RunRecord.turns is deliberately NOT
        # guarded: CPython's list.append is atomic and the SSE reader only
        # reads by growing index, so locking the per-turn hot path buys nothing.
        self._lock = threading.RLock()
        self._captures_dir = captures_dir
        self._registry = registry or RunRegistry(captures_dir / "results.db")

    def _make_record(self, cfg: RunConfig, label: str, tags: list[str] | None = None) -> RunRecord:
        run_id = uuid.uuid4().hex[:8]
        safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", label)
        jsonl = self._captures_dir / f"web_{safe}_{run_id}.jsonl"
        rconf = cfg.slot_config("retrieval")
        retrieval = cfg.plugin_name("retrieval") + (
            "+relational" if rconf.get("relational") else ""
        )
        # Goal encoding for the record: an item code, "level>=N" (character level),
        # "<skill>_level>=N" (skill level), or "state:<key>" (World self-reports
        # solved-ness, e.g. HanoiWorld's "solved") -- decoded back in _persist.
        goal = str(cfg.run.get("goal_item") or "")
        if not goal and cfg.run.get("goal_level") is not None:
            skill = str(cfg.run.get("goal_skill") or "")
            goal = f"{skill + '_' if skill else ''}level>={cfg.run.get('goal_level')}"
        if not goal and cfg.run.get("goal_state_key"):
            goal = f"state:{cfg.run['goal_state_key']}"
        if not goal and cfg.run.get("goal_monster"):
            goal = f"monster:{cfg.run['goal_monster']}"
        record = RunRecord(
            id=run_id,
            config_name=label,
            task=cfg.task,
            plugins={slot: cfg.plugin_name(slot) for slot in kernel.SLOTS},
            # Each World IS a benchmark in this dual-benchmark design (herobench /
            # hanoi), so the world plugin name doubles as the Reports filter key.
            benchmark=cfg.plugin_name("world"),
            model=str(cfg.slot_config("decoder").get("model") or ""),
            strategy=cfg.plugin_name("prompt"),
            retrieval=retrieval,
            goal=goal,
            max_turns=cfg.max_turns,
            capture_path=str(jsonl),
            started_at=_now(),
            tags=tags or [],
        )
        with self._lock:
            self._runs[run_id] = record
        return record

    def _run_record(self, record: RunRecord, cfg: RunConfig, stop: threading.Event) -> None:
        """Build the loop for an already-registered record and play it (blocking).
        A build failure (missing scoped creds, absent driver) must mark the record
        errored -- an uncaught raise here would leave it 'running' forever and, in
        a batch, silently skip the trial."""
        with self._lock:
            self._stops[record.id] = stop
        capture = _WebCapture(record, Path(record.capture_path))
        try:
            loop = _build_loop(cfg, capture)
        except Exception as exc:
            record.status = "error"
            record.error = str(exc)
            self._persist(record)
            return
        self._play(loop, cfg.max_turns, record, stop)

    def start(self, cfg: RunConfig, label: str, tags: list[str] | None = None) -> RunRecord:
        record = self._make_record(cfg, label, tags=tags)
        # Preflight the loop build NOW (in the caller), so a construction failure
        # surfaces as the API's clean 400 rather than a background run that starts
        # 200 and errors out of sight.
        capture = _WebCapture(record, Path(record.capture_path))
        try:
            loop = _build_loop(cfg, capture)
        except Exception:
            with self._lock:
                self._runs.pop(record.id, None)  # don't leave a phantom record behind
            raise
        stop = threading.Event()
        with self._lock:
            self._stops[record.id] = stop
        thread = threading.Thread(
            target=self._play, args=(loop, cfg.max_turns, record, stop), daemon=True
        )
        thread.start()
        return record

    def start_trials(self, cfg: RunConfig, label: str, n: int, temperature: float) -> str:
        """Run n stochastic trials of cfg sequentially in one background thread,
        resetting the HeroBench character before each so every trial starts fresh.
        Trial i uses seed=i (per-seed reproducible); all share a batch tag for
        Reports aggregation. Returns the batch id."""
        batch_id = uuid.uuid4().hex[:8]
        # A live-server reset only applies to HeroBench (a shared, stateful world);
        # a synthetic in-memory world like Hanoi resets itself by construction (a
        # fresh plugin instance is built per trial) -- calling the HeroBench reset
        # unconditionally would blind-POST delete/create at whatever character
        # happens to be configured (default character_1), corrupting an unrelated
        # HeroBench run/character that may be live on a different batch.
        is_herobench = cfg.plugin_name("world") == "herobench"
        wconf = cfg.slot_config("world")
        world_url = str(wconf.get("base_url", "http://127.0.0.1:8000"))
        character = str(wconf.get("character", "character_1"))
        batch_stop = threading.Event()
        with self._lock:
            self._batch_stops[batch_id] = batch_stop

        def run_batch() -> None:
            for i in range(1, n + 1):
                if batch_stop.is_set():  # batch aborted -> don't start the next trial
                    break
                if is_herobench:
                    with contextlib.suppress(Exception):  # a reset failure must not wedge the batch
                        reset_herobench_character(world_url, character)
                trial_cfg = copy.deepcopy(cfg)
                trial_cfg.slots["decoder"]["sampler"] = _stochastic_sampler(temperature, i)
                record = self._make_record(
                    trial_cfg,
                    f"{label}-t{i}",
                    # seed + temp fully reproduce the trial (top_k/top_p are fixed constants)
                    tags=[f"batch:{batch_id}", f"seed:{i}", f"temp:{temperature}"],
                )
                trial_stop = threading.Event()
                if batch_stop.is_set():  # aborted during setup -> end this trial at once
                    trial_stop.set()
                self._run_record(record, trial_cfg, trial_stop)  # blocking, sequential

        threading.Thread(target=run_batch, daemon=True).start()
        return batch_id

    def stop_batch(self, batch_id: str) -> bool:
        """Abort a whole trial batch: no further trials start, and the in-flight
        trial ends after its current turn."""
        with self._lock:
            ev = self._batch_stops.get(batch_id)
            if ev is None:
                return False
            ev.set()  # Event.set is thread-safe; fine to do under the RLock
            tag = f"batch:{batch_id}"
            for run_id, rec in list(self._runs.items()):
                if tag in rec.tags and rec.status == "running":
                    self.stop(run_id)  # re-entrant: stop() retakes the RLock
        return True

    def stop(self, run_id: str) -> bool:
        """Request a cooperative stop; the loop ends after its current turn."""
        with self._lock:
            ev = self._stops.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _play(
        self, loop: AgentLoop, max_turns: int, record: RunRecord, stop: threading.Event
    ) -> None:
        try:
            loop.play(max_turns, should_stop=stop.is_set)
            record.status = "stopped" if stop.is_set() else "done"
        except Exception as exc:  # surface harness/world/decoder failures to the UI
            record.status = "error"
            record.error = str(exc)
        finally:
            self._persist(record)

    def _persist(self, record: RunRecord) -> None:
        """Write the finished run to the registry (metrics from analyze)."""
        goal_item: str | None = None
        goal_level: int | None = None
        goal_skill: str | None = None
        goal_state_key: str | None = None
        goal_monster: str | None = None
        m = re.match(r"^(?:([a-z_]+)_)?level>=(\d+)$", record.goal)
        if record.goal.startswith("state:"):
            goal_state_key = record.goal[len("state:") :]
        elif record.goal.startswith("monster:"):
            goal_monster = record.goal[len("monster:") :]
        elif m:
            goal_skill, goal_level = m.group(1), int(m.group(2))
        elif record.goal:
            goal_item = record.goal
        metrics = analyze.analyze_turns(
            record.config_name,
            record.turns,
            goal_item=goal_item,
            goal_level=goal_level,
            goal_skill=goal_skill,
            goal_state_key=goal_state_key,
            goal_monster=goal_monster,
        )
        # the model under test may be ambient (not in the config); recover it from the
        # turns' recorded model id if so.
        model = record.model or metrics.model or ""
        with contextlib.suppress(Exception):  # reporting must never break a run
            self._registry.record(
                {
                    "id": record.id,
                    "benchmark": record.benchmark,
                    "model": model,
                    "strategy": record.strategy,
                    "retrieval": record.retrieval or metrics.backend,
                    "task": record.task,
                    "goal": record.goal,
                    "max_turns": record.max_turns,
                    "started_at": record.started_at,
                    "finished_at": _now(),
                    "status": record.status,
                    "metrics": analyze.metrics_as_dicts([metrics])[0],
                    "capture_path": record.capture_path,
                    "tags": record.tags,
                }
            )

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RunRecord]:
        with self._lock:  # snapshot under the lock; callers iterate it freely
            return list(self._runs.values())
