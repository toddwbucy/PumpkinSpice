"""Tower of Hanoi: a synthetic, vocabulary-disjoint planning domain.

Built as an independent cross-domain probe (see the harness-information-
starvation project note): HeroBench's "level a skill before attempting a gated
recipe" result is confounded by shared vocabulary/genre priors with anything we
teach it in-corpus. Hanoi shares NONE of that -- no crafting, no skills, no RPG
words -- so a model's execution-fidelity edge under the Stage-4 executor (or any
prompt strategy) either shows up here too, independent of any corpus content, or
it does not. Its rules are also universal (never place a larger disk on a
smaller one), so they belong directly in the system prompt; this world needs no
retrieval corpus at all (``retrieval="null"``), which removes the "did the KB
leak the answer" confound at the root rather than just weakening it.

Pure in-memory state machine, no HTTP. A fresh instance is constructed per run
(the plugin loader builds one per ``AgentLoop``), so THAT is the reset -- no
external reset call is needed between stochastic trials, unlike HeroBench's
shared live character.
"""

from __future__ import annotations

from typing import Any

from ..contracts import Action, ActionResult, WorldState

_PEGS = ("A", "B", "C")


class HanoiWorld:
    name = "hanoi"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.disks = int(config.get("disks", 4))
        if self.disks < 1:
            raise ValueError("hanoi: disks must be >= 1")
        # Largest disk first, so pegs[p][-1] is always the top (smallest) disk.
        self.pegs: dict[str, list[int]] = {"A": list(range(self.disks, 0, -1)), "B": [], "C": []}
        self.moves = 0

    def _solved_stack(self) -> list[int]:
        return list(range(self.disks, 0, -1))

    def _state(self) -> dict[str, Any]:
        return {
            "pegs": {p: list(self.pegs[p]) for p in _PEGS},
            "moves": self.moves,
            "disks": self.disks,
            "optimal_moves": 2**self.disks - 1,
            "solved": self.pegs["C"] == self._solved_stack(),
        }

    def get_state(self) -> WorldState:
        return WorldState(raw=self._state(), source=self.name)

    def act(self, action: Action) -> ActionResult:
        if action.kind != "move":
            return ActionResult(
                ok=False,
                status_code=400,
                error=f"unknown verb {action.kind!r}; only 'move' is legal",
            )
        src = str(action.args.get("from", "")).strip().upper()
        dst = str(action.args.get("to", "")).strip().upper()
        if src not in _PEGS or dst not in _PEGS:
            return ActionResult(
                ok=False,
                status_code=400,
                error=f"pegs must be one of A/B/C, got from={src!r} to={dst!r}",
            )
        if src == dst:
            return ActionResult(
                ok=False, status_code=422, error="source and destination peg are the same"
            )
        if not self.pegs[src]:
            return ActionResult(
                ok=False, status_code=422, error=f"peg {src} is empty, no disk to move"
            )
        disk = self.pegs[src][-1]
        if self.pegs[dst] and self.pegs[dst][-1] < disk:
            return ActionResult(
                ok=False,
                status_code=422,
                error=(
                    f"illegal move: disk {disk} is larger than the top disk on "
                    f"peg {dst} ({self.pegs[dst][-1]})"
                ),
            )
        self.pegs[src].pop()
        self.pegs[dst].append(disk)
        self.moves += 1
        return ActionResult(ok=True, status_code=200, data=self._state())
