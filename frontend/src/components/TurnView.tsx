import { useState } from "react";
import { Turn } from "../api";

export function TurnView({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false);
  const pos = `(${String(turn.world_state?.x ?? "?")}, ${String(turn.world_state?.y ?? "?")})`;
  const ok = turn.outcome?.ok;
  const nodes = turn.retrieval?.nodes ?? [];
  return (
    <div className="turn">
      <div className="meta">
        <strong>turn {turn.index}</strong> · at {pos} ·{" "}
        <span className="mono">
          {turn.action?.kind} {JSON.stringify(turn.action?.args ?? {})}
        </span>{" "}
        →{" "}
        <span className={`pill ${ok ? "ok" : "bad"}`}>
          {ok ? "ok" : `fail ${turn.outcome?.status_code ?? ""}`}
        </span>
        {turn.retrieval?.backend && (
          <span className="muted">
            {" "}
            · {turn.retrieval.backend} {Math.round(turn.retrieval.latency_ms ?? 0)}ms
          </span>
        )}
        <button className="ghost" style={{ marginLeft: 10, padding: "2px 8px" }} onClick={() => setOpen(!open)}>
          {open ? "hide" : "details"}
        </button>
      </div>
      {nodes.length > 0 && (
        <ul className="nodes">
          {nodes.slice(0, 5).map((n, i) => (
            <li key={i}>
              <span className="mono">[{n.score.toFixed(3)}]</span> {n.id}
            </li>
          ))}
        </ul>
      )}
      {open && (
        <div style={{ marginTop: 8 }}>
          <label>Raw model output</label>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", background: "var(--cream)", padding: 8, borderRadius: 8 }}>
            {turn.raw_output || "(none)"}
          </pre>
        </div>
      )}
    </div>
  );
}
