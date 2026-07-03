import { useEffect, useRef, useState } from "react";
import { ConfigInfo, Turn, getConfigs, runEventStream, startRun } from "../api";
import { TurnView } from "./TurnView";

export function Runs() {
  const [configs, setConfigs] = useState<ConfigInfo[]>([]);
  const [selected, setSelected] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [status, setStatus] = useState<string>("");
  const [err, setErr] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []); // cancel the stream on unmount

  useEffect(() => {
    getConfigs()
      .then((c) => {
        setConfigs(c);
        if (c.length) setSelected(c[0].name);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  async function launch() {
    if (!selected) return;
    setErr("");
    setTurns([]);
    setStatus("running");
    try {
      const { id } = await startRun(selected);
      abortRef.current?.abort(); // supersede any previous stream
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      await runEventStream(
        id,
        (t) => setTurns((prev) => [...prev, t]),
        (s, e) => {
          setStatus(s);
          if (e) setErr(e);
        },
        ctrl.signal,
      );
    } catch (e) {
      setErr(String(e));
      setStatus("error");
    }
  }

  const cfg = configs.find((c) => c.name === selected);

  return (
    <>
      <div className="card">
        <div className="row">
          <div style={{ flex: 1 }}>
            <label>Config</label>
            <br />
            <select
              style={{ minWidth: 260 }}
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              {configs.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} — {c.retrieval ?? "?"} / {c.world ?? "?"}
                </option>
              ))}
            </select>
          </div>
          <button className="btn" disabled={status === "running" || !selected} onClick={() => void launch()}>
            {status === "running" ? "Running…" : "Launch run"}
          </button>
          {status && (
            <span className={`pill ${status === "done" ? "ok" : status === "error" ? "bad" : "run"}`}>
              {status}
            </span>
          )}
        </div>
        {cfg?.task && <div className="muted" style={{ marginTop: 8 }}>Task: {cfg.task}</div>}
        {err && <div className="err" style={{ marginTop: 8 }}>{err}</div>}
      </div>

      <div className="card">
        {turns.length === 0 && <div className="muted">Launch a run to watch the agent play turn by turn.</div>}
        {turns.map((t) => (
          <TurnView key={t.index} turn={t} />
        ))}
      </div>
    </>
  );
}
