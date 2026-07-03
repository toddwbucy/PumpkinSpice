import { useEffect, useMemo, useRef, useState } from "react";
import { MapTile, Turn, getRuns, getWorldMap, runEventStream, stopBatch, stopRun } from "../api";

const HEROBENCH_URL = "http://127.0.0.1:8000";
const CELL = 22;

const CONTENT_COLORS: Record<string, string> = {
  monster: "#e06b6b",
  resource: "#6fae6f",
  workshop: "#6f8fd0",
  bank: "#e0b34a",
  grand_exchange: "#c98bd0",
  tasks_master: "#d0a06f",
};

function tileColor(type?: string): string {
  return (type && CONTENT_COLORS[type]) || "#efe3d0";
}

function num(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

// Pure viewer: watches the currently-running (or last-launched) HeroBench run.
// Runs are launched from the Setup tab; this tab reattaches to whatever is running.
export function World() {
  const [tiles, setTiles] = useState<MapTile[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [cursor, setCursor] = useState(0);
  const [follow, setFollow] = useState(true);
  const [status, setStatus] = useState("");
  const [runId, setRunId] = useState("");
  const [stopping, setStopping] = useState(false);
  const [batch, setBatch] = useState("");
  const [stoppingBatch, setStoppingBatch] = useState(false);
  const [err, setErr] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  function watch(id: string) {
    abortRef.current?.abort(); // supersede any previous stream
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setTurns([]);
    setCursor(0);
    setFollow(true);
    setStatus("running");
    setRunId(id);
    setStopping(false);
    runEventStream(
      id,
      (t) => setTurns((prev) => [...prev, t]),
      (s, e) => {
        setStatus(s);
        if (e) setErr(e);
      },
      ctrl.signal,
    ).catch((e) => {
      setErr(String(e));
      setStatus("error");
    });
  }

  useEffect(() => {
    getWorldMap(HEROBENCH_URL).then(setTiles).catch(() => undefined);
    // Reattach to the latest running run (survives tab switches / page refresh).
    getRuns()
      .then((rs) => {
        const running = [...rs].reverse().find((r) => r.status === "running");
        if (running) {
          watch(running.id);
          const bt = (running.tags || []).find((t) => t.startsWith("batch:"));
          setBatch(bt ? bt.slice("batch:".length) : "");
        }
      })
      .catch(() => undefined);
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (follow && turns.length) setCursor(turns.length - 1);
  }, [turns, follow]);

  const turn: Turn | undefined = turns[cursor];
  const bounds = useMemo(() => {
    if (!tiles.length) return null;
    const xs = tiles.map((t) => t.x);
    const ys = tiles.map((t) => t.y);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }, [tiles]);

  const px = num(turn?.world_state?.x);
  const py = num(turn?.world_state?.y);
  const lastDecodeS = turns.length
    ? Math.round((turns[turns.length - 1].timings_ms?.decode ?? 0) / 1000)
    : 0;

  return (
    <>
      <div className="card">
        <div className="row">
          {!status && <span className="muted">No active run -- launch one from the Setup tab.</span>}
          {status && (
            <span className={`pill ${status === "done" ? "ok" : status === "error" ? "bad" : "run"}`}>{status}</span>
          )}
          {status === "running" && (
            <span className="muted" style={{ fontSize: 12 }}>
              {turns.length} turns · decoding turn {turns.length}…
              {lastDecodeS ? ` (~${lastDecodeS}s/turn)` : ""}
            </span>
          )}
          {status === "running" && runId && (
            <button
              className="ghost"
              style={{ marginLeft: batch ? undefined : "auto" }}
              disabled={stopping}
              onClick={() => {
                setStopping(true);
                stopRun(runId).catch(() => setStopping(false));
              }}
            >
              {stopping ? "stopping after this turn…" : batch ? "Stop trial" : "Stop run"}
            </button>
          )}
          {status === "running" && batch && (
            <button
              className="ghost"
              style={{ marginLeft: "auto" }}
              disabled={stoppingBatch}
              onClick={() => {
                setStoppingBatch(true);
                stopBatch(batch).catch(() => setStoppingBatch(false));
              }}
              title={`abort the whole batch ${batch}`}
            >
              {stoppingBatch ? "stopping batch…" : "Stop batch"}
            </button>
          )}
        </div>
        {err && <div className="err" style={{ marginTop: 8 }}>{err}</div>}
        {turns.length > 0 && (
          <div className="row" style={{ marginTop: 10 }}>
            <label>Turn {turn?.index ?? 0}</label>
            <input
              type="range"
              min={0}
              max={Math.max(0, turns.length - 1)}
              value={cursor}
              style={{ flex: 1 }}
              onChange={(e) => {
                setFollow(false);
                setCursor(Number(e.target.value));
              }}
            />
            <button className={`ghost ${follow ? "active" : ""}`} onClick={() => setFollow(!follow)}>
              {follow ? "following live" : "follow live"}
            </button>
          </div>
        )}
      </div>

      <div className="row" style={{ alignItems: "flex-start" }}>
        <div className="card" style={{ flexShrink: 0 }}>
          {!bounds && <div className="muted">Map loads when HeroBench is running ({HEROBENCH_URL}).</div>}
          {bounds && (
            <svg
              width={(bounds.maxX - bounds.minX + 1) * CELL}
              height={(bounds.maxY - bounds.minY + 1) * CELL}
              style={{ display: "block" }}
            >
              {tiles.map((t) => (
                <rect
                  key={`${t.x},${t.y}`}
                  x={(t.x - bounds.minX) * CELL}
                  y={(bounds.maxY - t.y) * CELL}
                  width={CELL - 1}
                  height={CELL - 1}
                  rx={3}
                  fill={tileColor(t.content?.type)}
                >
                  <title>
                    ({t.x},{t.y}) {t.content ? `${t.content.type}: ${t.content.code}` : "empty"}
                  </title>
                </rect>
              ))}
              {Number.isFinite(px) && Number.isFinite(py) && (
                <circle
                  cx={(px - bounds.minX) * CELL + CELL / 2}
                  cy={(bounds.maxY - py) * CELL + CELL / 2}
                  r={CELL * 0.38}
                  fill="#2b1d12"
                  stroke="#fff"
                  strokeWidth={2}
                />
              )}
            </svg>
          )}
          <div className="row" style={{ gap: 12, marginTop: 8, fontSize: 12 }}>
            {Object.entries(CONTENT_COLORS).map(([k, v]) => (
              <span key={k} className="muted">
                <span style={{ display: "inline-block", width: 10, height: 10, background: v, borderRadius: 2, marginRight: 4 }} />
                {k}
              </span>
            ))}
          </div>
        </div>

        <div className="card" style={{ flex: 1 }}>
          {!turn && <div className="muted">Launch a run (Setup tab) to watch the agent move through the world.</div>}
          {turn && (
            <>
              <Stats turn={turn} />
              {turn.plan && (
                <div style={{ marginTop: 10 }}>
                  <label>Committed plan</label>
                  <pre
                    className="mono"
                    style={{
                      whiteSpace: "pre-wrap",
                      background: "var(--cream-2)",
                      padding: 8,
                      borderRadius: 8,
                      borderLeft: "3px solid var(--accent)",
                    }}
                  >
                    {turn.plan}
                  </pre>
                </div>
              )}
              <div style={{ marginTop: 10 }}>
                <label>Action</label>
                <div className="mono">
                  {turn.action.kind} {JSON.stringify(turn.action.args)} -&gt;{" "}
                  <span className={`pill ${turn.outcome.ok ? "ok" : "bad"}`}>
                    {turn.outcome.ok ? "ok" : `fail ${turn.outcome.status_code}`}
                  </span>
                </div>
              </div>
              {turn.reasoning && (
                <div style={{ marginTop: 10 }}>
                  <label>Reasoning trace</label>
                  <div className="thinking">{turn.reasoning}</div>
                </div>
              )}
              <div style={{ marginTop: 10 }}>
                <label>Raw output</label>
                <pre className="mono" style={{ whiteSpace: "pre-wrap", background: "var(--cream)", padding: 8, borderRadius: 8 }}>
                  {turn.raw_output || "(empty)"}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

function Stats({ turn }: { turn: Turn }) {
  const s = turn.world_state;
  const hp = num(s.hp);
  const maxHp = num(s.max_hp);
  const pct = Number.isFinite(hp) && Number.isFinite(maxHp) && maxHp > 0 ? (hp / maxHp) * 100 : 0;
  const stat = (label: string, key: string) => (
    <span className="muted" style={{ marginRight: 16 }}>
      {label}: <strong style={{ color: "var(--ink)" }}>{String(s[key] ?? "?")}</strong>
    </span>
  );
  return (
    <div>
      <div className="row" style={{ gap: 0 }}>
        {stat("level", "level")}
        {stat("xp", "xp")}
        {stat("gold", "gold")}
        <span className="muted">
          pos: <strong style={{ color: "var(--ink)" }}>({String(s.x ?? "?")}, {String(s.y ?? "?")})</strong>
        </span>
      </div>
      <div style={{ marginTop: 6 }}>
        <span className="muted" style={{ fontSize: 12 }}>
          HP {String(s.hp ?? "?")}
          {Number.isFinite(maxHp) ? ` / ${maxHp}` : ""}
        </span>
        <div style={{ height: 8, background: "var(--cream-2)", borderRadius: 4, overflow: "hidden", marginTop: 2 }}>
          <div style={{ width: `${pct}%`, height: "100%", background: pct > 30 ? "var(--ok)" : "var(--bad)" }} />
        </div>
      </div>
    </div>
  );
}
