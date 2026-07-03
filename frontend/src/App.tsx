import { useEffect, useState } from "react";
import { LoadedModel, getLoadedModels } from "./api";
import { Model } from "./components/Model";
import { Playground } from "./components/Playground";
import { Setup } from "./components/Setup";
import { World } from "./components/World";
import { Runs } from "./components/Runs";
import { Reports } from "./components/Reports";
import { Settings } from "./components/Settings";
import { HanoiSetup } from "./components/HanoiSetup";
import { HanoiBoard } from "./components/HanoiBoard";

type Top = "model" | "chat" | "herobench" | "hanoi" | "settings";
type HBTab = "setup" | "world" | "runs" | "reports";
type HanoiTab = "setup" | "board" | "reports";

function LoadedBadge() {
  const [models, setModels] = useState<LoadedModel[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      getLoadedModels()
        .then((m) => alive && (setModels(m), setErr(false)))
        .catch(() => alive && setErr(true));
    poll();
    const id = setInterval(poll, 4000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  if (err) return <span className="badge badge-off">decoder unreachable</span>;
  if (models.length === 0) return <span className="badge badge-off">no model loaded</span>;
  const m = models[0];
  return (
    <span className="badge badge-on" title={models.map((x) => x.id).join(", ")}>
      ● {m.id}
      {m.quantization ? ` · ${m.quantization}` : ""}
      {m.loaded_context_length ? ` · ${m.loaded_context_length.toLocaleString()} ctx` : ""}
      {models.length > 1 ? ` (+${models.length - 1})` : ""}
    </span>
  );
}

export function App() {
  const [top, setTop] = useState<Top>("model");
  const [hb, setHb] = useState<HBTab>("setup");
  const [hn, setHn] = useState<HanoiTab>("setup");

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">🎃</span>
        <span className="title">
          Pumpkin<span className="spice">Spice</span>
        </span>
        <LoadedBadge />
      </header>

      <nav className="tabs">
        <button className={top === "model" ? "active" : ""} onClick={() => setTop("model")}>
          Model
        </button>
        <button className={top === "chat" ? "active" : ""} onClick={() => setTop("chat")}>
          Chat
        </button>
        <button className={top === "herobench" ? "active" : ""} onClick={() => setTop("herobench")}>
          🎃 HeroBench
        </button>
        <button className={top === "hanoi" ? "active" : ""} onClick={() => setTop("hanoi")}>
          🗼 Hanoi
        </button>
        <button className={top === "settings" ? "active" : ""} onClick={() => setTop("settings")}>
          Settings
        </button>
      </nav>

      {top === "model" && <Model />}
      {top === "chat" && <Playground />}
      {top === "herobench" && (
        <>
          <nav className="subtabs">
            <button className={hb === "setup" ? "active" : ""} onClick={() => setHb("setup")}>
              Setup
            </button>
            <button className={hb === "world" ? "active" : ""} onClick={() => setHb("world")}>
              World
            </button>
            <button className={hb === "runs" ? "active" : ""} onClick={() => setHb("runs")}>
              Runs
            </button>
            <button className={hb === "reports" ? "active" : ""} onClick={() => setHb("reports")}>
              Reports
            </button>
          </nav>

          {hb === "setup" && <Setup onLaunched={() => setHb("world")} />}
          {hb === "world" && <World />}
          {hb === "runs" && <Runs />}
          {hb === "reports" && <Reports benchmark="herobench" />}
        </>
      )}
      {top === "hanoi" && (
        <>
          <nav className="subtabs">
            <button className={hn === "setup" ? "active" : ""} onClick={() => setHn("setup")}>
              Setup
            </button>
            <button className={hn === "board" ? "active" : ""} onClick={() => setHn("board")}>
              Board
            </button>
            <button className={hn === "reports" ? "active" : ""} onClick={() => setHn("reports")}>
              Reports
            </button>
          </nav>

          {hn === "setup" && <HanoiSetup onLaunched={() => setHn("board")} />}
          {hn === "board" && <HanoiBoard />}
          {hn === "reports" && <Reports benchmark="hanoi" />}
        </>
      )}
      {top === "settings" && <Settings />}
    </div>
  );
}
