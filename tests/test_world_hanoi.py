"""Tower of Hanoi world: legality rules, state shape, and a full solve."""

from __future__ import annotations

import pytest

from pumpkinspice.contracts import Action
from pumpkinspice.plugins.world_hanoi import HanoiWorld


def test_initial_state() -> None:
    w = HanoiWorld({"disks": 3})
    s = w.get_state().raw
    assert s["pegs"] == {"A": [3, 2, 1], "B": [], "C": []}
    assert s["moves"] == 0 and s["disks"] == 3
    assert s["optimal_moves"] == 7  # 2^3 - 1
    assert s["solved"] is False


def test_disks_config_validated() -> None:
    with pytest.raises(ValueError):
        HanoiWorld({"disks": 0})


def test_legal_move_updates_state() -> None:
    w = HanoiWorld({"disks": 2})
    r = w.act(Action(kind="move", args={"from": "A", "to": "C"}))
    assert r.ok is True and r.status_code == 200
    assert w.pegs == {"A": [2], "B": [], "C": [1]}
    assert w.moves == 1
    assert r.data["pegs"] == {"A": [2], "B": [], "C": [1]}  # post-action state, for _state_after


def test_illegal_larger_on_smaller_rejected() -> None:
    w = HanoiWorld({"disks": 2})
    w.act(Action(kind="move", args={"from": "A", "to": "C"}))  # disk 1 -> C
    r = w.act(Action(kind="move", args={"from": "A", "to": "C"}))  # disk 2 onto disk 1
    assert r.ok is False and r.status_code == 422
    assert "larger" in (r.error or "")
    assert w.moves == 1  # rejected move did not mutate state


def test_empty_peg_and_same_peg_and_unknown_verb_and_bad_peg_rejected() -> None:
    w = HanoiWorld({"disks": 2})
    assert w.act(Action(kind="move", args={"from": "B", "to": "C"})).status_code == 422  # empty
    assert w.act(Action(kind="move", args={"from": "A", "to": "A"})).status_code == 422  # same peg
    assert w.act(Action(kind="fight", args={})).status_code == 400  # unknown verb
    assert w.act(Action(kind="move", args={"from": "A", "to": "Z"})).status_code == 400  # bad peg


def test_full_solve_reaches_solved_in_optimal_moves() -> None:
    """Classic recursive solution for N=3: verifies solved flips true and the
    optimal move count (7) is exactly what the solver used."""

    def hanoi(n: int, src: str, aux: str, dst: str, moves: list[tuple[str, str]]) -> None:
        if n == 0:
            return
        hanoi(n - 1, src, dst, aux, moves)
        moves.append((src, dst))
        hanoi(n - 1, aux, src, dst, moves)

    w = HanoiWorld({"disks": 3})
    moves: list[tuple[str, str]] = []
    hanoi(3, "A", "B", "C", moves)
    assert len(moves) == 7
    for src, dst in moves:
        r = w.act(Action(kind="move", args={"from": src, "to": dst}))
        assert r.ok is True
    s = w.get_state().raw
    assert s["solved"] is True and s["moves"] == 7 == s["optimal_moves"]
    assert s["pegs"]["C"] == [3, 2, 1]
