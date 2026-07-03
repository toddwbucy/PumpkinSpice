import { useEffect, useMemo, useState } from "react";
import {
  LeaderboardEntry,
  RunMetrics,
  RunReport,
  getLeaderboard,
  getReportRuns,
  updateReportRun,
} from "../api";

type RunSortKey =
  | "model"
  | "goal"
  | "strategy"
  | "steps"
  | "success"
  | "failed_actions"
  | "revisits"
  | "replans"
  | "decode_tok_s";

// Sortable value for a run+column. Strings sort lexically; missing numbers sort last.
function runSortVal(r: RunReport, key: RunSortKey): number | string {
  if (key === "model") return r.model || "";
  if (key === "goal") return r.goal || "";
  if (key === "strategy") return r.strategy || "";
  const m = r.metrics || {};
  if (key === "success") return m.success === true ? 1 : m.success === false ? 0 : -1;
  const v = m[key as keyof RunMetrics];
  return typeof v === "number" ? v : Number.NEGATIVE_INFINITY;
}

function okMark(s: boolean | null | undefined) {
  if (s === true) return <span className="pill ok">yes</span>;
  if (s === false) return <span className="pill bad">no</span>;
  return <span className="pill">?</span>;
}

function RunRow({ r, onSaved }: { r: RunReport; onSaved: () => void }) {
  const [label, setLabel] = useState(r.label || "");
  const [tags, setTags] = useState((r.tags || []).join(", "));
  const m = r.metrics || {};

  async function save() {
    const tagList = tags.split(",").map((t) => t.trim()).filter(Boolean);
    await updateReportRun(r.id, { label, tags: tagList });
    onSaved();
  }

  return (
    <tr>
      <td title={r.model}>{r.model || "(ambient)"}</td>
      <td title={r.task}>{r.goal || "-"}</td>
      <td>{r.strategy}</td>
      <td className="num">{m.steps ?? "-"}</td>
      <td>{okMark(m.success as boolean | null)}</td>
      <td className="num">{m.failed_actions ?? "-"}</td>
      <td className="num">{m.revisits ?? "-"}</td>
      <td className="num">{m.replans ?? "-"}</td>
      <td className="num">{m.decode_tok_s != null ? (m.decode_tok_s as number).toFixed(1) : "-"}</td>
      <td>
        <input
          className="cell-edit"
          value={label}
          placeholder="label"
          onChange={(e) => setLabel(e.target.value)}
          onBlur={save}
          onKeyDown={(e) => e.key === "Enter" && save()}
        />
      </td>
      <td>
        <input
          className="cell-edit"
          value={tags}
          placeholder="tag, tag"
          onChange={(e) => setTags(e.target.value)}
          onBlur={save}
          onKeyDown={(e) => e.key === "Enter" && save()}
        />
      </td>
    </tr>
  );
}

interface BatchSummary {
  batch: string;
  task: string;
  strategy: string;
  n: number;
  completed: number;
  stopped: number;
  errored: number;
  avgSteps: number | null;
}

export function Reports({ benchmark = "herobench" }: { benchmark?: string }) {
  const [mode, setMode] = useState<"runs" | "batches" | "leaderboard">("runs");
  const [runs, setRuns] = useState<RunReport[]>([]);
  const [board, setBoard] = useState<LeaderboardEntry[]>([]);
  const [sortKey, setSortKey] = useState<RunSortKey>("model");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [err, setErr] = useState("");

  function reload() {
    getReportRuns({ benchmark }).then(setRuns).catch((e) => setErr(String(e)));
    getLeaderboard(benchmark).then(setBoard).catch(() => undefined);
  }
  useEffect(reload, [benchmark]);

  const sortedRuns = useMemo(() => {
    const arr = [...runs];
    arr.sort((a, b) => {
      const va = runSortVal(a, sortKey);
      const vb = runSortVal(b, sortKey);
      const cmp =
        typeof va === "string" && typeof vb === "string"
          ? va.localeCompare(vb)
          : (va as number) - (vb as number);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [runs, sortKey, sortDir]);

  function toggleSort(key: RunSortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      // text columns default A->Z; numeric columns default high->low (most interesting first)
      setSortDir(key === "model" || key === "goal" || key === "strategy" ? "asc" : "desc");
    }
  }

  // A clickable, sort-aware header cell.
  const th = (label: string, k: RunSortKey, num = false) => (
    <th className={`sortable${num ? " num" : ""}`} onClick={() => toggleSort(k)}>
      {label}
      {sortKey === k ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
    </th>
  );

  // Group trial runs by their batch:<id> tag and aggregate (stochastic-trial view).
  const batches = useMemo<BatchSummary[]>(() => {
    const byBatch: Record<string, RunReport[]> = {};
    for (const r of runs) {
      const bt = (r.tags || []).find((t) => t.startsWith("batch:"));
      if (bt) (byBatch[bt] ||= []).push(r);
    }
    return Object.entries(byBatch).map(([bt, rs]) => {
      const steps = rs
        .map((r) => r.metrics?.steps)
        .filter((s): s is number => typeof s === "number");
      return {
        batch: bt.slice("batch:".length),
        task: rs[0]?.task ?? "",
        strategy: rs[0]?.strategy ?? "",
        n: rs.length,
        completed: rs.filter((r) => r.metrics?.success === true).length,
        stopped: rs.filter((r) => r.status === "stopped").length,
        errored: rs.filter((r) => r.status === "error").length,
        avgSteps: steps.length ? steps.reduce((a, b) => a + b, 0) / steps.length : null,
      };
    });
  }, [runs]);

  return (
    <>
      <div className="card row">
        {(["runs", "batches", "leaderboard"] as const).map((mo) => (
          <button key={mo} className={`ghost ${mode === mo ? "active" : ""}`} onClick={() => setMode(mo)}>
            {mo === "batches" ? `Batches (${batches.length})` : mo[0].toUpperCase() + mo.slice(1)}
          </button>
        ))}
        <span className="muted" style={{ fontSize: 12 }}>
          {runs.length} runs
        </span>
        <button className="ghost" style={{ marginLeft: "auto" }} onClick={reload}>
          refresh
        </button>
      </div>

      {err && <div className="card err">{err}</div>}

      <div className="card">
        {mode === "runs" && (
          <table className="rtable">
            <thead>
              <tr>
                {th("model", "model")}
                {th("goal", "goal")}
                {th("strategy", "strategy")}
                {th("steps", "steps", true)}
                {th("ok", "success")}
                {th("fail", "failed_actions", true)}
                {th("revis", "revisits", true)}
                {th("replan", "replans", true)}
                {th("tok/s", "decode_tok_s", true)}
                <th>label</th>
                <th>tags</th>
              </tr>
            </thead>
            <tbody>
              {sortedRuns.map((r) => (
                <RunRow key={r.id} r={r} onSaved={reload} />
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={11} className="muted">
                    No runs yet. Launch one from HeroBench &rarr; Setup, or import captures with
                    <code> pumpkinspice reports-import</code>.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}

        {mode === "batches" && (
          <table className="rtable">
            <thead>
              <tr>
                <th>batch</th>
                <th>task</th>
                <th>strategy</th>
                <th className="num">trials</th>
                <th className="num">completed</th>
                <th className="num">stopped</th>
                <th className="num">errored</th>
                <th className="num">avg steps</th>
              </tr>
            </thead>
            <tbody>
              {batches.map((b) => (
                <tr key={b.batch}>
                  <td className="mono">{b.batch}</td>
                  <td title={b.task}>{b.task.slice(0, 36)}</td>
                  <td>{b.strategy}</td>
                  <td className="num">{b.n}</td>
                  <td className="num">
                    {b.completed}/{b.n}
                  </td>
                  <td className="num">{b.stopped}</td>
                  <td className="num">{b.errored}</td>
                  <td className="num">{b.avgSteps != null ? b.avgSteps.toFixed(1) : "-"}</td>
                </tr>
              ))}
              {batches.length === 0 && (
                <tr>
                  <td colSpan={8} className="muted">
                    No trial batches yet. Set Trials &gt; 1 in HeroBench &rarr; Setup to run a batch.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}

        {mode === "leaderboard" && (
          <table className="rtable">
            <thead>
              <tr>
                <th>#</th>
                <th>model</th>
                <th className="num">runs</th>
                <th className="num">success</th>
                <th className="num">rate</th>
                <th className="num">best steps</th>
                <th className="num">avg steps</th>
              </tr>
            </thead>
            <tbody>
              {board.map((b, i) => (
                <tr key={b.model}>
                  <td className="num">{i + 1}</td>
                  <td title={b.model}>{b.model || "(ambient)"}</td>
                  <td className="num">{b.runs}</td>
                  <td className="num">{b.successes}</td>
                  <td className="num">{(b.success_rate * 100).toFixed(0)}%</td>
                  <td className="num">{b.best_steps ?? "--"}</td>
                  <td className="num">{b.avg_steps != null ? b.avg_steps.toFixed(1) : "--"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
