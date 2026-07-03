"""RunManager thread safety: its _runs/_stops/_batch_stops dicts are hit
concurrently by API handler threads and background run/batch threads. Hammer
registration, listing, and stop lookups from many threads and assert nothing
raises and no record is lost. (The per-turn RunRecord.turns append path is
deliberately lock-free -- see the RunManager lock comment -- and is exercised
by the run-lifecycle tests, not here.)"""

from __future__ import annotations

import threading
from pathlib import Path

from pumpkinspice.config import load_config
from pumpkinspice.web.runs import RunManager

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

N_THREADS = 8
RECORDS_PER_THREAD = 25


def test_concurrent_record_registration_list_and_stop(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "offline.toml")
    mgr = RunManager(tmp_path)
    errors: list[BaseException] = []
    barrier = threading.Barrier(N_THREADS)

    def hammer(idx: int) -> None:
        try:
            barrier.wait(timeout=5.0)  # line the threads up for maximum overlap
            for i in range(RECORDS_PER_THREAD):
                mgr._make_record(cfg, f"t{idx}-{i}", tags=[f"batch:b{idx}", f"seed:{i}"])
                mgr.list()  # snapshot must not blow up mid-registration
                assert mgr.get(f"unknown-{idx}-{i}") is None
                assert mgr.stop(f"unknown-{idx}-{i}") is False
                assert mgr.stop_batch(f"unknown-batch-{idx}") is False
        except BaseException as exc:  # pragma: no cover - only fires on a regression
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert len(mgr._runs) == N_THREADS * RECORDS_PER_THREAD
    assert len(mgr.list()) == N_THREADS * RECORDS_PER_THREAD
