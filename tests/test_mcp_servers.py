"""MCP server registry (the in-UI manager's persistence)."""

from __future__ import annotations

from pathlib import Path

from pumpkinspice.web.mcp_servers import McpServer, McpServerStore


def test_empty_when_absent(tmp_path: Path) -> None:
    assert McpServerStore(tmp_path / "mcp.json").all() == []


def test_upsert_replaces_by_name_and_persists(tmp_path: Path) -> None:
    store = McpServerStore(tmp_path / "mcp.json")
    store.upsert(McpServer(name="memory", command="npx", args=["-y", "@mcp/server-memory"]))
    store.upsert(McpServer(name="fs", command="npx", args=["-y", "@mcp/server-fs", "/tmp"]))
    # same name -> replace, not duplicate
    store.upsert(McpServer(name="memory", command="uvx", args=["mcp-memory"]))
    servers = McpServerStore(tmp_path / "mcp.json").all()  # reopen -> persisted
    assert {s.name for s in servers} == {"memory", "fs"}
    mem = next(s for s in servers if s.name == "memory")
    assert mem.command == "uvx" and mem.args == ["mcp-memory"]


def test_enable_and_delete(tmp_path: Path) -> None:
    store = McpServerStore(tmp_path / "mcp.json")
    store.upsert(McpServer(name="memory", command="npx"))
    assert store.set_enabled("memory", False) is True
    assert store.all()[0].enabled is False
    assert store.set_enabled("nope", True) is False  # unknown name
    assert store.delete("memory") is True
    assert store.all() == []
    assert store.delete("memory") is False  # already gone


def test_malformed_entries_skipped(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text('[{"name": "ok", "command": "npx"}, {"name": "nocommand"}, "junk"]')
    names = [s.name for s in McpServerStore(p).all()]
    assert names == ["ok"]  # entries without a command (or non-dicts) are dropped
