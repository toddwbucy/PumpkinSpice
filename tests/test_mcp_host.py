"""MCP host pure logic (tool spec formatting, result flattening, call routing).
The live stdio connection is integration-verified against a real MCP server."""

from __future__ import annotations

import asyncio
import dataclasses

from pumpkinspice.web.mcp_host import McpHost, _result_text, _tool_spec


def test_tool_spec_namespaces_and_defaults() -> None:
    spec = _tool_spec(
        "memory",
        "search",
        "find things",
        {"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "memory__search"  # namespaced by server
    assert spec["function"]["description"] == "find things"
    assert spec["function"]["parameters"]["properties"]["q"]["type"] == "string"
    # no schema -> a minimal object schema (so the tool is still callable)
    assert _tool_spec("s", "t", "", None)["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


@dataclasses.dataclass
class _Block:
    text: str


@dataclasses.dataclass
class _Result:
    content: list[_Block]
    isError: bool = False


def test_result_text_flattens_and_marks_errors() -> None:
    assert _result_text(_Result([_Block("hello"), _Block("world")])) == "hello\nworld"
    assert _result_text(_Result([])) == "(no output)"
    assert _result_text(_Result([_Block("boom")], isError=True)) == "error: boom"


def test_call_unknown_tool_is_graceful() -> None:
    # an empty host (no servers entered) routes an unknown tool to a clean error,
    # never an exception that would kill the chat loop
    host = McpHost([])
    assert asyncio.run(host.call("nope__x", {})) == "error: unknown tool nope__x"
