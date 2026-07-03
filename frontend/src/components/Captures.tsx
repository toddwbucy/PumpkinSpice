import { useEffect, useState } from "react";
import { CaptureInfo, Turn, getCapture, getCaptures } from "../api";
import { TurnView } from "./TurnView";

export function Captures() {
  const [files, setFiles] = useState<CaptureInfo[]>([]);
  const [selected, setSelected] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [err, setErr] = useState("");

  const refresh = () =>
    getCaptures()
      .then(setFiles)
      .catch((e) => setErr(String(e)));

  useEffect(() => {
    void refresh();
  }, []);

  function open(name: string) {
    setSelected(name);
    setErr("");
    getCapture(name)
      .then(setTurns)
      .catch((e) => setErr(String(e)));
  }

  return (
    <div className="row" style={{ alignItems: "flex-start" }}>
      <div className="card" style={{ width: 280, flexShrink: 0 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <label>Capture files</label>
          <button className="ghost" onClick={() => void refresh()}>
            ↻
          </button>
        </div>
        {files.length === 0 && <div className="muted">No captures yet.</div>}
        {files.map((f) => (
          <div
            key={f.name}
            className={`list-item ${f.name === selected ? "active" : ""}`}
            onClick={() => open(f.name)}
          >
            <span className="mono" style={{ fontSize: 12 }}>
              {f.name}
            </span>
            <span className="muted">{f.turns}</span>
          </div>
        ))}
      </div>

      <div className="card" style={{ flex: 1 }}>
        {err && <div className="err">{err}</div>}
        {!selected && <div className="muted">Select a capture to inspect its turns.</div>}
        {turns.map((t) => (
          <TurnView key={t.index} turn={t} />
        ))}
      </div>
    </div>
  );
}
