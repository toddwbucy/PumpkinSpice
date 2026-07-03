"""Stage-1 metrics: per-run analysis over capture turns."""

from __future__ import annotations

from pumpkinspice.analyze import analyze_turns, comparison_table


def _turn(
    idx: int,
    *,
    x: int,
    y: int,
    level: int,
    xp: int,
    kind: str,
    ok: bool,
    inv: dict | None = None,
    model: str = "m1",
    backend: str = "pgvector",
    empty: bool = False,
) -> dict:
    data = {"level": level, "xp": xp, "x": x, "y": y}
    if inv is not None:
        data["inventory"] = inv
    return {
        "index": idx,
        "world_state": {"level": level, "xp": xp, "x": x, "y": y},
        "retrieval": {"backend": backend},
        "action": {"kind": kind, "args": {}},
        "outcome": {"ok": ok, "status_code": 200 if ok else 489, "data": data},
        "timings_ms": {"decode": 50.0, "retrieval": 10.0},
        "decoder_empty": empty,
        "model": model,
    }


def test_metrics_outcome_and_progress() -> None:
    turns = [
        _turn(0, x=0, y=0, level=1, xp=0, kind="move", ok=True),
        _turn(1, x=1, y=0, level=1, xp=10, kind="fight", ok=True),
        _turn(2, x=1, y=0, level=1, xp=10, kind="move", ok=False),  # failed (revisit pos)
        _turn(
            3, x=0, y=0, level=2, xp=60, kind="rest", ok=True, empty=True, inv={"copper_dagger": 1}
        ),
    ]
    m = analyze_turns("run.jsonl", turns, goal_item="copper_dagger")
    assert m.model == "m1" and m.backend == "pgvector"
    assert m.steps == 4
    assert m.failed_actions == 1
    assert m.no_ops == 1
    assert m.success is True  # copper_dagger in final inventory
    assert m.level_delta == 1  # 1 -> 2
    assert m.xp_delta == 60
    assert m.revisits == 2  # turn 2 revisits (1,0); turn 3 revisits (0,0)
    assert m.action_counts == {"move": 2, "fight": 1, "rest": 1}


def test_token_throughput_metrics() -> None:
    # 100 + 200 completion tokens over 1000ms + 1000ms decode = 2.0s
    turns = [
        {
            **_turn(0, x=0, y=0, level=1, xp=0, kind="move", ok=True),
            "completion_tokens": 100,
            "timings_ms": {"decode": 1000.0},
        },
        {
            **_turn(1, x=1, y=0, level=1, xp=0, kind="move", ok=True),
            "completion_tokens": 200,
            "timings_ms": {"decode": 1000.0},
        },
    ]
    m = analyze_turns("r", turns)
    assert m.avg_gen_tokens == 150.0  # (100 + 200) / 2
    assert m.decode_tok_s == 150.0  # 300 tokens / 2.0s
    assert "tok/s" in comparison_table([m])


def test_success_requires_crafting_not_residual() -> None:
    # A reset character carrying a residual dagger must NOT read as success; only an
    # increase (it was crafted this run) counts.
    def t(i: int, qty: int) -> dict:
        inv = [{"code": "copper_dagger", "quantity": qty}]
        return {
            "index": i,
            "world_state": {"level": 1, "xp": 0, "x": 0, "y": 0, "inventory": inv},
            "retrieval": {},
            "action": {"kind": "rest", "args": {}},
            "outcome": {"ok": True, "data": {"level": 1, "x": 0, "y": 0, "inventory": inv}},
            "timings_ms": {"decode": 1.0},
        }

    # present from the start, unchanged -> NOT success
    assert analyze_turns("r", [t(0, 1), t(1, 1)], goal_item="copper_dagger").success is False
    # gained one (crafted) -> success
    assert analyze_turns("r", [t(0, 1), t(1, 2)], goal_item="copper_dagger").success is True


def test_success_from_nested_craft_response() -> None:
    # the completing craft turn nests the updated character (with the new dagger)
    # under "character" -- success must unwrap it, not read the empty top level.
    first = {
        "index": 0,
        "world_state": {"level": 1, "x": 0, "y": 0, "inventory": []},
        "retrieval": {},
        "action": {"kind": "move", "args": {}},
        "outcome": {"ok": True, "data": {}},
        "timings_ms": {"decode": 1.0},
    }
    craft = {
        "index": 1,
        "world_state": {"level": 1, "x": 2, "y": 1, "inventory": []},
        "retrieval": {},
        "action": {"kind": "craft", "args": {"code": "copper_dagger"}},
        "outcome": {
            "ok": True,
            "data": {
                "character": {
                    "level": 1,
                    "x": 2,
                    "y": 1,
                    "inventory": [{"code": "copper_dagger", "quantity": 1}],
                }
            },
        },
        "timings_ms": {"decode": 1.0},
    }
    assert analyze_turns("r", [first, craft], goal_item="copper_dagger").success is True


def test_goal_level_and_failure() -> None:
    turns = [_turn(0, x=0, y=0, level=1, xp=0, kind="move", ok=True)]
    assert analyze_turns("r", turns, goal_level=3).success is False
    assert analyze_turns("r", turns, goal_level=1).success is True
    assert analyze_turns("r", turns).success is None  # no goal -> unknown


def test_comparison_table_sorts_success_first() -> None:
    a = analyze_turns(
        "a",
        [
            _turn(
                0,
                x=0,
                y=0,
                level=1,
                xp=0,
                kind="rest",
                ok=True,
                model="big",
                inv={"copper_dagger": 1},
            )
        ],
        goal_item="copper_dagger",
    )
    b = analyze_turns(
        "b",
        [_turn(0, x=0, y=0, level=1, xp=0, kind="rest", ok=True, model="small")],
        goal_item="copper_dagger",
    )
    table = comparison_table([b, a])
    assert "model" in table and "steps" in table
    # the successful run (big) should be listed before the unsuccessful (small)
    assert table.index("big") < table.index("small")
