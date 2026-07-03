import { useEffect, useState } from "react";
import { getPromptOptions, getRetrievalOptions, startRunOptions, startTrials } from "../api";

const STRATEGY_LABELS: Record<string, string> = {
  default: "reactive (Stage 1)",
  plan: "plan -> execute (Stage 2)",
  replan: "plan + replan (Stage 3)",
  executor: "plan executor (Stage 4: harness holds the plan)",
};

// Presets carry the task, the stop-on-craft goal item, and a sensible turn budget,
// so picking one fills all three (you can still override any field).
interface TaskPreset {
  label: string;
  task: string;
  goal: string;
  turns: number;
}
const TASK_PRESETS: TaskPreset[] = [
  { label: "Craft a copper dagger (L1, baseline)", task: "Craft a copper dagger.", goal: "copper_dagger", turns: 50 },
  {
    label: "Level weaponcrafting to 5 (calibrated: winnable, no combat)",
    task: "Reach weaponcrafting level 5.",
    goal: "weaponcrafting_level>=5",
    turns: 100, // the gather->smelt->craft triangle costs ~15 turns/dagger at 1-tile moves
  },
  {
    label: "Craft a sticky sword (L5: combat + level gate; combat-confounded)",
    task: "Craft a sticky sword (needs weaponcrafting level 5; fight 2 yellow slimeballs).",
    goal: "sticky_sword",
    turns: 100,
  },
  { label: "Reach character level 5 by fighting", task: "Reach character level 5 by fighting.", goal: "", turns: 60 },
  { label: "Gather 10 ash wood", task: "Gather 10 ash wood.", goal: "", turns: 30 },
  { label: "Defeat the lich (L30 at (9,7))", task: "Defeat the lich (level 30 at (9,7)).", goal: "", turns: 80 },
];
const CUSTOM = "__custom__";

export function Setup({ onLaunched }: { onLaunched: () => void }) {
  const [retrievalOptions, setRetrievalOptions] = useState<string[]>([]);
  const [retrieval, setRetrieval] = useState("pgvector+relational");
  const [promptOptions, setPromptOptions] = useState<string[]>([]);
  const [strategy, setStrategy] = useState("default");
  const [preset, setPreset] = useState(TASK_PRESETS[0].label);
  const [task, setTask] = useState(TASK_PRESETS[0].task);
  const [maxTurns, setMaxTurns] = useState(TASK_PRESETS[0].turns);
  const [goalItem, setGoalItem] = useState(TASK_PRESETS[0].goal);
  const [trials, setTrials] = useState(1);
  const [temp, setTemp] = useState(0.7);
  const [seed, setSeed] = useState(""); // single-run: blank = greedy, set = reproduce that seed
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    getRetrievalOptions().then(setRetrievalOptions).catch((e) => setErr(String(e)));
    getPromptOptions().then(setPromptOptions).catch(() => undefined);
  }, []);

  function pickPreset(label: string) {
    setPreset(label);
    if (label === CUSTOM) return; // keep current task/goal/turns; let them edit
    const p = TASK_PRESETS.find((x) => x.label === label);
    if (p) {
      setTask(p.task);
      setGoalItem(p.goal);
      setMaxTurns(p.turns);
    }
  }

  async function launch() {
    if (!task.trim()) return;
    setErr("");
    setBusy(true);
    try {
      if (trials > 1) {
        // N stochastic trials (temp/top_k released), fresh reset between each
        await startTrials(retrieval, task, strategy, maxTurns, goalItem.trim(), trials, temp);
      } else {
        // single run: a seed reproduces that stochastic trajectory; blank = greedy
        const seedNum = seed.trim() ? Number(seed) : undefined;
        await startRunOptions(retrieval, task, strategy, maxTurns, goalItem.trim(), seedNum, temp);
      }
      onLaunched(); // hand off to the World tab to watch it play
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h3 style={{ marginTop: 0 }}>Configure a HeroBench run</h3>
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
          <label>Retrieval / database</label>
          <br />
          <select value={retrieval} onChange={(e) => setRetrieval(e.target.value)}>
            {retrievalOptions.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Max turns</label>
          <br />
          <input
            type="number"
            style={{ width: 80 }}
            value={maxTurns}
            onChange={(e) => setMaxTurns(Math.max(1, Number(e.target.value) || 50))}
          />
        </div>
        <div>
          <label>Goal (item, or skill_level&gt;=N)</label>
          <br />
          <input
            style={{ width: 180 }}
            value={goalItem}
            placeholder="sticky_sword | weaponcrafting_level>=5"
            onChange={(e) => setGoalItem(e.target.value)}
            title="Stop-on-goal: an item code stops on craft; level>=N or <skill>_level>=N stops on reaching that level; blank runs to max turns"
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
      <div style={{ marginTop: 12 }}>
        <label>Task / objective</label>
        <br />
        <select style={{ width: "100%" }} value={preset} onChange={(e) => pickPreset(e.target.value)}>
          {TASK_PRESETS.map((p) => (
            <option key={p.label} value={p.label}>
              {p.label}
            </option>
          ))}
          <option value={CUSTOM}>Custom task...</option>
        </select>
        {preset === CUSTOM && (
          <input
            style={{ width: "100%", marginTop: 8 }}
            value={task}
            placeholder="Type a task, e.g. Craft a fire staff."
            onChange={(e) => setTask(e.target.value)}
          />
        )}
        {preset !== CUSTOM && (
          <div className="muted mono" style={{ marginTop: 6, fontSize: 12 }}>
            {task}
          </div>
        )}
      </div>
      <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
        Runs the model shown in the header on the live HeroBench world. The agent is the controlled
        conventional-RAG control -- plain vector search, no tools. Each turn takes ~5-60s by model.
        {trials > 1 && (
          <>
            {" "}
            <strong>{trials} trials</strong> run sequentially at temperature {temp} (seed 1..{trials},
            reproducible), with a fresh character reset between each; grouped under one batch in
            Reports.
          </>
        )}
        {trials === 1 && seed.trim() !== "" && (
          <>
            {" "}
            <strong>Reproducing seed {seed}</strong> at temperature {temp} -- the same stochastic
            sampler the trials use, so this run recreates that exact trajectory (tagged seed/temp).
          </>
        )}
      </div>
      {err && <div className="err" style={{ marginTop: 8 }}>{err}</div>}
      <div style={{ marginTop: 12 }}>
        <button className="btn" disabled={busy || !task.trim()} onClick={() => void launch()}>
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
