import { useEffect, useRef, useState } from "react";
import { Turn, getRuns, runEventStream, stopBatch, stopRun } from "../api";

const DISK_COLORS = ["#e06b6b", "#e0b34a", "#6fae6f", "#6f8fd0", "#c98bd0", "#d0a06f", "#7ecbd6", "#b08cd8"];

function pegsOf(state: Record<string, unknown>): Record<string, number[]> {
  const p = state.pegs;
  if (!p || typeof p !== "object") return { A: [], B: [], C: [] };
  const out: Record<string, number[]> = {};
  for (const k of ["A", "B", "C"]) {
    const v = (p as Record<string, unknown>)[k];
    out[k] = Array.isArray(v) ? (v as number[]) : [];
  }
  return out;
}

function num(v: unknown): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

// Pure viewer: watches the currently-running (or last-launched) Hanoi run, same
// reattach-on-mount pattern as World.tsx. Renders the 3 pegs as stacked disks.
export function HanoiBoard() {
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
    abortRef.current?.abort();
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
    getRuns()
      .then((rs) => {
        const running = [...rs].reverse().find((r) => r.status === "running" && r.config_name.includes("hanoi"));
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
  const state = turn?.world_state ?? {};
  const pegs = pegsOf(state);
  const disks = num(state.disks) ?? Math.max(...Object.values(pegs).flat(), 0);
  const moves = num(state.moves);
  const optimal = num(state.optimal_moves);
  const solved = state.solved === true;

  return (
    <>
      <div className="card">
        <div className="row">
          {!status && <span className="muted">No active Hanoi run -- launch one from the Setup tab.</span>}
          {status && (
            <span className={`pill ${status === "done" ? "ok" : status === "error" ? "bad" : "run"}`}>{status}</span>
          )}
          {status === "running" && (
            <span className="muted" style={{ fontSize: 12 }}>
              {turns.length} turns · {moves ?? 0} legal moves so far
              {optimal ? ` (optimal: ${optimal})` : ""}
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
        <div className="card" style={{ flexShrink: 0, minWidth: 320 }}>
          {!turn && <div className="muted">Launch a run (Setup tab) to watch the pegs.</div>}
          {turn && (
            <>
              <div className="row" style={{ justifyContent: "space-around", alignItems: "flex-end", height: 160 }}>
                {["A", "B", "C"].map((p) => (
                  <div key={p} style={{ display: "flex", flexDirection: "column-reverse", alignItems: "center", gap: 2 }}>
                    {pegs[p].map((d, i) => (
                      <div
                        key={i}
                        title={`disk ${d}`}
                        style={{
                          width: disks ? 20 + (d / disks) * 80 : 40,
                          height: 14,
                          background: DISK_COLORS[(d - 1) % DISK_COLORS.length],
                          borderRadius: 3,
                        }}
                      />
                    ))}
                    <div style={{ width: 100, height: 4, background: "var(--ink)", borderRadius: 2, marginTop: 2 }} />
                    <div className="muted" style={{ marginTop: 4 }}>
                      peg {p}
                    </div>
                  </div>
                ))}
              </div>
              <div className="row" style={{ marginTop: 12, gap: 16 }}>
                <span className="muted">
                  moves: <strong style={{ color: "var(--ink)" }}>{moves ?? "?"}</strong>
                  {optimal ? ` / optimal ${optimal}` : ""}
                </span>
                <span className={`pill ${solved ? "ok" : ""}`}>{solved ? "solved!" : "not solved"}</span>
              </div>
            </>
          )}
        </div>

        <div className="card" style={{ flex: 1 }}>
          {!turn && <div className="muted">Launch a run (Setup tab) to watch the agent move disks.</div>}
          {turn && (
            <>
              {turn.plan && (
                <div style={{ marginTop: 0 }}>
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
