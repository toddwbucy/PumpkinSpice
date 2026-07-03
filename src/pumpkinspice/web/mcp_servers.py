"""MCP server registry for the in-UI server manager (web PRD Phase 3a).

Persists the operator's MCP server list (name + stdio launch command + args +
enabled) to ``captures/mcp_servers.json``. The MCP HOST (3a-ii) spawns the enabled
servers over stdio via the `mcp` SDK and aggregates their tools for Chat; this module
is just the config CRUD, with no SDK dependency (so the manager works even before the
host is wired). Chat-only -- the benchmark agent never touches these.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class McpServer:
    name: str
    command: str  # e.g. "npx" or "python"
    args: list[str] = dataclasses.field(default_factory=list)
    enabled: bool = True


class McpServerStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[McpServer]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
        except (ValueError, OSError):
            return []
        out: list[McpServer] = []
        for d in raw if isinstance(raw, list) else []:
            if isinstance(d, dict) and d.get("name") and d.get("command"):
                out.append(
                    McpServer(
                        name=str(d["name"]),
                        command=str(d["command"]),
                        args=[str(a) for a in d.get("args", [])],
                        enabled=bool(d.get("enabled", True)),
                    )
                )
        return out

    def _save(self, servers: list[McpServer]) -> None:
        self.path.write_text(json.dumps([dataclasses.asdict(s) for s in servers], indent=2))

    def upsert(self, server: McpServer) -> McpServer:
        """Add a server, or replace the one with the same name."""
        servers = [s for s in self.all() if s.name != server.name]
        servers.append(server)
        self._save(servers)
        return server

    def delete(self, name: str) -> bool:
        servers = self.all()
        kept = [s for s in servers if s.name != name]
        if len(kept) == len(servers):
            return False
        self._save(kept)
        return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        servers = self.all()
        found = False
        for s in servers:
            if s.name == name:
                s.enabled = enabled
                found = True
        if found:
            self._save(servers)
        return found
