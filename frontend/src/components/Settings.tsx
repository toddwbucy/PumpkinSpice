import { useEffect, useState } from "react";
import {
  McpServer,
  deleteMcpServer,
  getApiToken,
  getMcpServers,
  setApiToken,
  setMcpEnabled,
  upsertMcpServer,
} from "../api";

// Bearer token sent on every /api call (Authorization header). Only needed when the
// server enforces auth via PUMPKINSPICE_API_TOKEN.
function ApiTokenCard() {
  const [token, setToken] = useState(getApiToken());
  const [saved, setSaved] = useState(false);

  function save() {
    setApiToken(token.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>API token</h3>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
        Required only when the server is started with PUMPKINSPICE_API_TOKEN (e.g. when exposed
        on the LAN). Leave blank for local unauthenticated use.
      </div>
      <div className="row">
        <div style={{ flex: 1, minWidth: 220 }}>
          <label>token</label>
          <br />
          <input
            type="password"
            style={{ width: "100%" }}
            value={token}
            placeholder="paste the server token"
            onChange={(e) => setToken(e.target.value)}
          />
        </div>
        <button className="btn" onClick={save}>
          Save
        </button>
        {saved && <span className="muted" style={{ fontSize: 12 }}>saved</span>}
      </div>
    </div>
  );
}

// Phase 3a: the in-UI MCP server manager. Servers PumpkinSpice spawns over stdio to
// give the Chat tab tools + memory. Chat-only -- the benchmark agent never uses these.
// (Phase 3b will add the chat-RAG backend config here too.)
export function Settings() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [name, setName] = useState("");
  const [command, setCommand] = useState("npx");
  const [args, setArgs] = useState("");
  const [err, setErr] = useState("");

  function reload() {
    getMcpServers().then(setServers).catch((e) => setErr(String(e)));
  }
  useEffect(reload, []);

  async function add() {
    if (!name.trim() || !command.trim()) return;
    setErr("");
    try {
      await upsertMcpServer({
        name: name.trim(),
        command: command.trim(),
        args: args.split(/\s+/).filter(Boolean),
        enabled: true,
      });
      setName("");
      setArgs("");
      reload();
    } catch (e) {
      setErr(String(e));
    }
  }
  async function toggle(s: McpServer) {
    try {
      await setMcpEnabled(s.name, !s.enabled);
      reload();
    } catch (e) {
      setErr(String(e));
    }
  }
  async function remove(n: string) {
    try {
      await deleteMcpServer(n);
      reload();
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <>
      <ApiTokenCard />

      <div className="card">
        <h3 style={{ marginTop: 0 }}>MCP servers</h3>
        <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
          Servers PumpkinSpice spawns over stdio to give Chat tools + memory. Chat-only -- the
          HeroBench benchmark agent never uses these (the fairness firewall).
        </div>
        {err && <div className="err" style={{ marginBottom: 8 }}>{err}</div>}
        <table className="rtable">
          <thead>
            <tr>
              <th>name</th>
              <th>command</th>
              <th>args</th>
              <th>enabled</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {servers.map((s) => (
              <tr key={s.name}>
                <td>{s.name}</td>
                <td className="mono">{s.command}</td>
                <td className="mono" style={{ fontSize: 12 }}>{s.args.join(" ")}</td>
                <td>
                  <button className={`ghost ${s.enabled ? "active" : ""}`} onClick={() => void toggle(s)}>
                    {s.enabled ? "on" : "off"}
                  </button>
                </td>
                <td>
                  <button className="ghost" onClick={() => void remove(s.name)}>
                    remove
                  </button>
                </td>
              </tr>
            ))}
            {servers.length === 0 && (
              <tr>
                <td colSpan={5} className="muted">
                  No MCP servers yet. Add one below (the Chat tool loop arrives next).
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Add a server</h3>
        <div className="row">
          <div>
            <label>name</label>
            <br />
            <input value={name} placeholder="memory" onChange={(e) => setName(e.target.value)} />
          </div>
          <div>
            <label>command</label>
            <br />
            <input value={command} placeholder="npx / uvx / python" onChange={(e) => setCommand(e.target.value)} />
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <label>args (space-separated)</label>
            <br />
            <input
              style={{ width: "100%" }}
              value={args}
              placeholder="-y @modelcontextprotocol/server-memory"
              onChange={(e) => setArgs(e.target.value)}
            />
          </div>
          <button className="btn" disabled={!name.trim() || !command.trim()} onClick={() => void add()}>
            Add
          </button>
        </div>
      </div>
    </>
  );
}
