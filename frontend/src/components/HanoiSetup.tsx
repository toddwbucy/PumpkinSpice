import { useEffect, useState } from "react";
import { getPromptOptions, startHanoiRun, startHanoiTrials } from "../api";

const STRATEGY_LABELS: Record<string, string> = {
  default: "reactive (Stage 1)",
  plan: "plan -> execute (Stage 2)",
  replan: "plan + replan (Stage 3)",
  executor: "plan executor (Stage 4: harness holds the plan)",
};

// Tower of Hanoi: a vocabulary-disjoint second benchmark. No corpus/retrieval
// composer needed (the rules are universal, stated directly in the system
// prompt) and no goal field (the goal is always "all disks on peg C") -- only
// the prompt strategy and puzzle size vary. See world_hanoi.py for why.
export function HanoiSetup({ onLaunched }: { onLaunched: () => void }) {
  const [promptOptions, setPromptOptions] = useState<string[]>([]);
  const [strategy, setStrategy] = useState("default");
  const [disks, setDisks] = useState(4);
  const [maxTurns, setMaxTurns] = useState(100);
  const [trials, setTrials] = useState(1);
  const [temp, setTemp] = useState(0.7);
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    getPromptOptions().then(setPromptOptions).catch(() => undefined);
  }, []);

  const optimalMoves = 2 ** disks - 1;

  async function launch() {
    setErr("");
    setBusy(true);
    try {
      if (trials > 1) {
        await startHanoiTrials(strategy, maxTurns, disks, trials, temp);
      } else {
        const seedNum = seed.trim() ? Number(seed) : undefined;
        await startHanoiRun(strategy, maxTurns, disks, seedNum, temp);
      }
      onLaunched();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Configure a Tower of Hanoi run</h3>
      <div className="row">
        <div>
          <label>Strategy</label>
          <br />
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {promptOptions.map((p) => (
              <option key={p} value={p}>
                {STRATEGY_LABELS[p] ?? p}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Disks</label>
          <br />
          <input
            type="number"
            min={1}
            max={8}
            style={{ width: 70 }}
            value={disks}
            onChange={(e) => setDisks(Math.max(1, Math.min(8, Number(e.target.value) || 1)))}
          />
        </div>
        <div>
          <label>Max turns</label>
          <br />
          <input
            type="number"
            style={{ width: 80 }}
            value={maxTurns}
            onChange={(e) => setMaxTurns(Math.max(1, Number(e.target.value) || 100))}
          />
        </div>
        <div>
          <label>Trials (1 = single run)</label>
          <br />
          <input
            type="number"
            min={1}
            max={50}
            style={{ width: 70 }}
            value={trials}
            onChange={(e) => setTrials(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
          />
        </div>
        {trials === 1 && (
          <div>
            <label>Seed (blank = greedy)</label>
            <br />
            <input
              style={{ width: 90 }}
              value={seed}
              placeholder="e.g. 7"
              onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
            />
          </div>
        )}
        {(trials > 1 || seed.trim() !== "") && (
          <div>
            <label>Temperature</label>
            <br />
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              style={{ width: 80 }}
              value={temp}
              onChange={(e) => setTemp(Math.max(0, Math.min(2, Number(e.target.value) || 0)))}
            />
          </div>
        )}
      </div>
      <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
        Move the entire {disks}-disk stack from peg A to peg C (never place a larger disk on a
        smaller one). Optimal solution: <strong>{optimalMoves} moves</strong>. A synthetic,
        vocabulary-disjoint domain -- no corpus, no game genre, an independent cross-domain probe
        of whether a strategy's plan-holding edge generalizes beyond HeroBench.
        {trials > 1 && (
          <>
            {" "}
            <strong>{trials} trials</strong> run sequentially at temperature {temp} (seed 1..
            {trials}, reproducible) -- a fresh puzzle instance every trial, no reset needed.
          </>
        )}
        {trials === 1 && seed.trim() !== "" && (
          <>
            {" "}
            <strong>Reproducing seed {seed}</strong> at temperature {temp}.
          </>
        )}
      </div>
      {err && <div className="err" style={{ marginTop: 8 }}>{err}</div>}
      <div style={{ marginTop: 12 }}>
        <button className="btn" disabled={busy} onClick={() => void launch()}>
          {busy
            ? "Launching..."
            : trials > 1
              ? `Run ${trials} trials`
              : seed.trim() !== ""
                ? `Reproduce seed ${seed}`
                : "Launch & watch"}
        </button>
      </div>
    </div>
  );
}
