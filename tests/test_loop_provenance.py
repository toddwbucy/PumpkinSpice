"""Decode provenance in the capture: each Turn records the request the decoder ACTUALLY
sent (Turn.decode = decoder.last_request), so runs are groupable by their IV (enable_thinking
no-think arm, sampler/seed, max_tokens length-cap) post-hoc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pumpkinspice import kernel
from pumpkinspice.loop import AgentLoop


class _SettingsDecoder:
    """Sets last_request like the real OpenAI-compat decoders do in complete()."""

    name = "settings"

    def __init__(self) -> None:
        self.last_request: dict[str, Any] = {}

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        self.last_request = {
            "temperature": 0,
            "seed": 7,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return '{"action": "rest", "args": {}}'


class _BareDecoder:
    name = "bare"

    def complete(self, prompt: str, *, sampler: dict[str, Any] | None = None) -> str:
        return '{"action": "rest", "args": {}}'


def _loop(decoder: object, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        decoder=decoder,  # type: ignore[arg-type]
        retrieval=kernel.load_plugin("retrieval", "null", {}),
        world=kernel.load_plugin("world", "mock", {}),
        prompt=kernel.load_plugin("prompt", "default", {}),
        capture=kernel.load_plugin("capture", "jsonl", {"path": str(tmp_path / "c.jsonl")}),
        task="t",
    )


def test_decode_records_the_request_as_sent(tmp_path: Path) -> None:
    turns = _loop(_SettingsDecoder(), tmp_path).play(1)
    dec = turns[0].decode
    assert dec["chat_template_kwargs"]["enable_thinking"] is False  # the no-think IV
    assert dec["max_tokens"] == 256  # the length-cap knob is recorded
    assert dec["seed"] == 7


def test_decode_empty_for_bare_decoder(tmp_path: Path) -> None:
    # a decoder that does not expose last_request (mock/echo) records empty, no crash
    turns = _loop(_BareDecoder(), tmp_path).play(1)
    assert turns[0].decode == {}
