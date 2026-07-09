"""Network plugins tested without live services via httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from pumpkinspice.contracts import Action
from pumpkinspice.plugins.decoder_lmstudio import LMStudioDecoder
from pumpkinspice.plugins.decoder_vllm import VLLMDecoder
from pumpkinspice.plugins.world_herobench import HeroBenchWorld


def _mock_client(handler, base_url: str) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)


def test_lmstudio_payload_and_parse() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    d = LMStudioDecoder({"base_url": "http://x", "model": "m", "max_tokens": 7})
    d._client = _mock_client(handler, "http://x")

    out = d.complete("hello", sampler={"temperature": 0.5})
    assert out == "hi"
    assert captured["model"] == "m"
    assert captured["max_tokens"] == 7
    assert captured["messages"][0]["content"] == "hello"
    assert captured["temperature"] == 0.5  # per-call override
    assert captured["top_k"] == 1  # greedy default retained


def test_lmstudio_max_tokens_default_and_unbounded() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    # unset -> unbounded (default) so a reasoning model finishes; cap omitted
    d = LMStudioDecoder({"base_url": "http://x"})
    d._client = _mock_client(handler, "http://x")
    d.complete("hi")
    assert "max_tokens" not in captured

    # an explicit cap is honored (e.g. to bound a rambling non-reasoning model)
    d256 = LMStudioDecoder({"base_url": "http://x", "max_tokens": 256})
    d256._client = _mock_client(handler, "http://x")
    d256.complete("hi")
    assert captured["max_tokens"] == 256


def test_enable_thinking_and_extra_body() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

    # enable_thinking=False -> vLLM chat_template_kwargs in the payload (the v2 no-think IV)
    d = LMStudioDecoder({"base_url": "http://x", "enable_thinking": False})
    d._client = _mock_client(handler, "http://x")
    d.complete("hi")
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}

    # a general extra_body passthrough merges into the payload; unset enable_thinking is
    # not sent (the internal-CoT baseline arm = the model's default).
    d2 = LMStudioDecoder({"base_url": "http://x", "extra_body": {"guided_choice": ["a", "b"]}})
    d2._client = _mock_client(handler, "http://x")
    d2.complete("hi")
    assert captured["guided_choice"] == ["a", "b"]
    assert "chat_template_kwargs" not in captured


def test_lmstudio_captures_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer", "reasoning_content": "thinking"}}]},
        )

    d = LMStudioDecoder({"base_url": "http://x"})
    d._client = _mock_client(handler, "http://x")
    assert d.complete("q") == "answer"
    assert d.last_reasoning == "thinking"  # chain-of-thought captured for the viewer/capture


def test_lmstudio_captures_gpt_oss_reasoning_field() -> None:
    # gpt-oss (harmony) returns its chain-of-thought under `reasoning`, not
    # `reasoning_content`; the decoder must fall back to it or the thinking is lost.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer", "reasoning": "gpt-oss cot"}}]},
        )

    d = LMStudioDecoder({"base_url": "http://x"})
    d._client = _mock_client(handler, "http://x")
    assert d.complete("q") == "answer"
    assert d.last_reasoning == "gpt-oss cot"


def test_lmstudio_failed_request_clears_stale_state() -> None:
    # A 400 (e.g. prompt over a small-context model's window) must NOT leave the
    # previous turn's reasoning/usage readable -- that would double-count tokens.
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "a", "reasoning": "cot"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 900},
            },
        )

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "context length exceeded"})

    d = LMStudioDecoder({"base_url": "http://x"})
    d._client = _mock_client(ok, "http://x")
    d.complete("q")
    assert d.last_reasoning == "cot" and d.last_usage["completion_tokens"] == 900

    d._client = _mock_client(bad, "http://x")
    with pytest.raises(httpx.HTTPStatusError):
        d.complete("too big")
    assert d.last_reasoning == ""
    assert d.last_usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_lmstudio_null_content_maps_to_empty() -> None:
    # a reasoning model mid-thought returns content: null -> must be "", not "None"
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    d = LMStudioDecoder({"base_url": "http://x"})
    d._client = _mock_client(handler, "http://x")
    assert d.complete("hello") == ""


def test_vllm_payload_uses_vllm_sampler_dialect() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    d = VLLMDecoder({"base_url": "http://x", "model": "qwen3-14b"})
    d._client = _mock_client(handler, "http://x")

    assert d.complete("hello") == "hi"
    assert captured["model"] == "qwen3-14b"  # required, always sent
    # greedy in vLLM's dialect, NOT llama.cpp's
    assert captured["top_k"] == -1  # -1 = all (llama.cpp would be 1)
    assert captured["repetition_penalty"] == 1.0  # not "repeat_penalty"
    assert "repeat_penalty" not in captured
    assert captured["temperature"] == 0


def test_vllm_requires_model() -> None:
    # vLLM 400s without a `model`; fail fast at construction with a clear message.
    with pytest.raises(ValueError, match="requires a 'model'"):
        VLLMDecoder({"base_url": "http://x"})


def test_vllm_default_port_avoids_herobench() -> None:
    # default must not collide with the HeroBench world server on :8000
    d = VLLMDecoder({"model": "m"})
    assert d.base_url == "http://127.0.0.1:8001"


def test_vllm_inherits_reasoning_and_null_handling() -> None:
    # the shared base logic (reasoning capture, null content -> "") reaches the subclass
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": None, "reasoning_content": "cot"}}]},
        )

    d = VLLMDecoder({"base_url": "http://x", "model": "m"})
    d._client = _mock_client(handler, "http://x")
    assert d.complete("q") == ""  # null content -> ""
    assert d.last_reasoning == "cot"  # reasoning captured via inherited logic


def test_herobench_get_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/characters/hero"
        return httpx.Response(200, json={"name": "hero", "level": 3})

    w = HeroBenchWorld({"base_url": "http://h", "character": "hero"})
    w._client = _mock_client(handler, "http://h")
    st = w.get_state()
    assert st.raw["level"] == 3
    assert st.source == "herobench"


def test_herobench_act_paths_aliases_and_body_shapes() -> None:
    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.content))
        return httpx.Response(200, json={"ok": True})

    w = HeroBenchWorld({"base_url": "http://h", "character": "hero"})
    w._client = _mock_client(handler, "http://h")

    assert w.act(Action(kind="move", args={"x": 1, "y": 2})).ok
    w.act(Action(kind="gather", args={"quantity": 3}))  # alias -> gathering
    w.act(Action(kind="fight", args={"quantity": 3}))  # -> /action/fight/3
    paths = [p for p, _ in seen]
    assert paths == [
        "/my/hero/action/move",
        "/my/hero/action/gathering",
        "/my/hero/action/fight/3",
    ]
    # multi-param move -> object body; single-scalar gathering -> bare value
    assert json.loads(seen[0][1]) == {"x": 1, "y": 2}
    assert json.loads(seen[1][1]) == 3  # NOT {"quantity": 3} -- FastAPI 422s on that


def test_herobench_error_status_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(489, json={"error": {"code": 489}})

    w = HeroBenchWorld({"base_url": "http://h"})
    w._client = _mock_client(handler, "http://h")
    r = w.act(Action(kind="move", args={"x": 99, "y": 99}))
    assert not r.ok
    assert r.status_code == 489
    assert r.error


def test_herobench_craft_failure_surfaces_reason() -> None:
    # HeroBench sends a craft-precondition failure as 500, detail DOUBLY nested at
    # body["error"]["message"]. The agent must see WHY (to Reflect), not just "HTTP 500".
    def craft_body(info: dict) -> dict:
        return {"error": {"code": 500, "message": info}}

    def make(info: dict) -> HeroBenchWorld:
        w = HeroBenchWorld({"base_url": "http://h"})
        w._client = _mock_client(lambda req: httpx.Response(500, json=craft_body(info)), "http://h")
        return w

    # wrong-tile (the "one tile short" case): surface where to go
    wrong_tile = make(
        {
            "errors": {"on_workshop_tile": False, "enough_items_for_craft": True},
            "workshop": {"needed": "(1, 5)", "current": "(1, 4)"},
        }
    )
    r = wrong_tile.act(Action(kind="craft", args={"code": "copper", "quantity": 6}))
    assert not r.ok and r.status_code == 500
    assert "go to the workshop at (1, 5)" in r.error and "(1, 4)" in r.error

    # insufficient ingredients: surface what is missing
    missing = make({"errors": {"on_workshop_tile": True}, "missing_items": {"copper_ore": 37}})
    r2 = missing.act(Action(kind="craft", args={"code": "copper", "quantity": 6}))
    assert "missing items" in r2.error and "copper_ore" in r2.error
    assert r2.data["error"]["message"]["missing_items"] == {"copper_ore": 37}  # body retained


def test_world_get_state_retries_once_then_succeeds(monkeypatch) -> None:
    """A single transient world failure must not abort a run: get_state retries
    once (after a short backoff) before giving up."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda s: None)  # no real backoff in tests
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "hiccup"})
        return httpx.Response(200, json={"x": 0, "y": 0, "level": 1})

    world = HeroBenchWorld({"base_url": "http://world", "character": "c1"})
    world._client = httpx.Client(base_url="http://world", transport=httpx.MockTransport(handler))
    state = world.get_state()
    assert state.raw["level"] == 1 and calls["n"] == 2  # failed once, retried, succeeded


def test_world_get_state_raises_with_context_after_retry(monkeypatch) -> None:
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    world = HeroBenchWorld({"base_url": "http://world", "character": "c1"})
    world._client = httpx.Client(base_url="http://world", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="after retry"):
        world.get_state()
