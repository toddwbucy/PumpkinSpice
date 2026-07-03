"""MCP host for the agentic Chat (web PRD Phase 3a-ii).

Opens stdio sessions to the enabled MCP servers for the duration of ONE chat request
(used as ``async with McpHost(servers) as host``), aggregates their tools in OpenAI
function-calling format (namespaced ``server__tool`` to avoid collisions), and routes
tool calls. Per-request lifetime keeps each session's anyio scopes inside a single
task, which sidesteps the cross-request session-lifetime pitfalls. Lazy-imports the
`mcp` SDK so the rest of the web app (incl. the server manager) works without the
optional `mcp` extra.

Chat-only: nothing here touches the benchmark agent loop (the fairness firewall).
"""

from __future__ import annotations

import contextlib
from typing import Any

from .mcp_servers import McpServer


def _require_mcp() -> tuple[Any, Any, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("MCP support not installed; run `uv sync --extra mcp`") from exc
    return ClientSession, StdioServerParameters, stdio_client


def _tool_spec(
    server_name: str, name: str, description: str, schema: dict[str, Any] | None
) -> dict[str, Any]:
    """One OpenAI-format tool entry, namespaced by server (pure; unit-tested)."""
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{name}",
            "description": description or "",
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


def _result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult's content blocks into text."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else str(block))
    out = "\n".join(parts) if parts else "(no output)"
    return f"error: {out}" if getattr(result, "isError", False) else out


class McpHost:
    def __init__(self, servers: list[McpServer]) -> None:
        self._servers = [s for s in servers if s.enabled]
        self._stack: contextlib.AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}  # server name -> ClientSession
        self._index: dict[str, tuple[str, str]] = {}  # ns tool name -> (server, raw name)
        self.tools: list[dict[str, Any]] = []  # aggregated, OpenAI format

    async def __aenter__(self) -> McpHost:  # pragma: no cover - spawns real servers
        client_session, stdio_params, stdio_client = _require_mcp()
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        for s in self._servers:
            read, write = await self._stack.enter_async_context(
                stdio_client(stdio_params(command=s.command, args=s.args))
            )
            session = await self._stack.enter_async_context(client_session(read, write))
            await session.initialize()
            self._sessions[s.name] = session
            for t in (await session.list_tools()).tools:
                self._index[f"{s.name}__{t.name}"] = (s.name, t.name)
                self.tools.append(_tool_spec(s.name, t.name, t.description or "", t.inputSchema))
        return self

    async def __aexit__(self, *exc: object) -> None:  # pragma: no cover
        if self._stack is not None:
            await self._stack.__aexit__(*exc)  # type: ignore[arg-type]
        self._stack = None

    async def call(self, ns_name: str, arguments: dict[str, Any]) -> str:
        if ns_name not in self._index:
            return f"error: unknown tool {ns_name}"
        server, raw = self._index[ns_name]
        try:
            result = await self._sessions[server].call_tool(raw, arguments)
        except Exception as exc:  # a tool error must not kill the chat loop
            return f"error calling {ns_name}: {exc}"
        return _result_text(result)
