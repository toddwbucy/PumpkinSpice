"""Decode provenance in the capture: each Turn records the effective sampler (incl. seed)
and extra_body actually sent, so runs are groupable by their IV (e.g. the enable_thinking
no-think arm) post-hoc."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pumpkinspice import kernel
from pumpkinspice.contracts import Action, ActionResult, WorldState
from pumpkinspice.loop import AgentLoop


class _World:
    name = "w"

    def get_state(self) -> WorldState:
        return WorldState(raw={"x": 0, "y": 0, "level": 1, "inventory": []})

    def act(self, action: Action) -> ActionResult:
        return ActionResult(ok=True, status_code=200)


class _SettingsDecoder:
    """Exposes decode settings like the real OpenAI-compat decoders do."""

    name = "settings"
    default_sampler: ClassVar[dict[str, Any]] = {"temperature": 0, "seed": 0}
    extra_body: ClassVar[dict[str, Any]] = {"chat_template_kwargs": {"enable_thinking": False}}

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "rest", "args": {}}'


class _BareDecoder:
    name = "bare"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "rest", "args": {}}'


def _loop(decoder: object, tmp_path: Path, **kw: Any) -> AgentLoop:
    return AgentLoop(
        decoder=decoder,  # type: ignore[arg-type]
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=_World(),  # type: ignore[arg-type]
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="t",
        **kw,
    )


def test_decode_provenance_records_sampler_and_extra_body(tmp_path: Path) -> None:
    # the per-call sampler override merges OVER the decoder's default (seed 7 wins)
    turns = _loop(_SettingsDecoder(), tmp_path, sampler={"seed": 7}).play(1)
    dec = turns[0].decode
    assert dec["sampler"] == {"temperature": 0, "seed": 7}
    assert dec["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_decode_provenance_empty_for_bare_decoder(tmp_path: Path) -> None:
    # a decoder that does not expose settings (mock/echo) records empty, no crash
    turns = _loop(_BareDecoder(), tmp_path).play(1)
    assert turns[0].decode == {"sampler": {}, "extra_body": {}}
