import { useEffect, useState } from "react";
import {
  LoadedModel,
  ModelSettings,
  getAvailableModels,
  getLoadedModels,
  getModelSettings,
  loadModel,
  updateModelSettings,
} from "../api";

function NumField({
  label,
  value,
  step,
  min,
  max,
  onSave,
}: {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  onSave: (v: number) => void;
}) {
  const [v, setV] = useState(String(value));
  useEffect(() => setV(String(value)), [value]);
  // clamp to [min, max] so an out-of-range value (e.g. a negative temperature,
  // which LMStudio rejects with a 400) can't be submitted
  const commit = () => {
    let n = Number(v) || 0;
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    setV(String(n));
    onSave(n);
  };
  return (
    <div>
      <label>{label}</label>
      <br />
      <input
        type="number"
        step={step ?? 1}
        min={min}
        max={max}
        style={{ width: 150 }}
        value={v}
        onChange={(e) => setV(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
      />
    </div>
  );
}

export function Model() {
  const [loaded, setLoaded] = useState<LoadedModel[]>([]);
  const [available, setAvailable] = useState<string[]>([]);
  const [settings, setSettings] = useState<ModelSettings | null>(null);
  const [pick, setPick] = useState("");
  const [loadingMsg, setLoadingMsg] = useState("");
  const [err, setErr] = useState("");

  function reloadLoaded() {
    getLoadedModels().then(setLoaded).catch(() => setLoaded([]));
  }
  useEffect(() => {
    reloadLoaded();
    getAvailableModels().then(setAvailable).catch(() => undefined);
    getModelSettings()
      .then((s) => {
        setSettings(s);
        setPick(s.model);
      })
      .catch((e) => setErr(String(e)));
    const id = setInterval(reloadLoaded, 4000);
    return () => clearInterval(id);
  }, []);

  async function setUnderTest(model: string) {
    setErr("");
    setSettings(await updateModelSettings({ model }));
    setPick(model);
  }
  async function load(model: string) {
    setErr("");
    setLoadingMsg(`Loading ${model}... (a large model can take ~60s)`);
    try {
      await loadModel(model);
      await setUnderTest(model);
      reloadLoaded();
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoadingMsg("");
    }
  }
  async function saveDefaults(patch: Partial<ModelSettings>) {
    setSettings(await updateModelSettings(patch));
  }

  return (
    <>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Model under test</h3>
        {loaded.length === 0 && <div className="muted">No model loaded in LMStudio.</div>}
        {loaded.map((m) => (
          <div key={m.id} className="row" style={{ alignItems: "baseline", marginBottom: 4 }}>
            <span className="badge badge-on">● {m.id}</span>
            <span className="muted" style={{ fontSize: 13 }}>
              {[
                m.arch,
                m.quantization,
                m.loaded_context_length ? `${m.loaded_context_length.toLocaleString()} ctx` : null,
              ]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </div>
        ))}

        <div className="row" style={{ marginTop: 12 }}>
          <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ minWidth: 240 }}>
            <option value="">(use whatever is loaded)</option>
            {available.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button className="ghost" disabled={pick === settings?.model} onClick={() => void setUnderTest(pick)}>
            Set as model under test
          </button>
          <button className="btn" disabled={!pick || !!loadingMsg} onClick={() => void load(pick)}>
            {loadingMsg ? "Loading..." : "Load now"}
          </button>
        </div>
        {loadingMsg && <div className="muted" style={{ marginTop: 6 }}>{loadingMsg}</div>}
        {settings?.model && (
          <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
            Model under test: <strong>{settings.model}</strong> -- used by Chat + benchmark runs
            (JIT-loads on first use).
          </div>
        )}
        {err && <div className="err" style={{ marginTop: 8 }}>{err}</div>}
      </div>

      {settings && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Decode defaults</h3>
          <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
            Applied to Chat and benchmark runs. 0 = unbounded / full history.
          </div>
          <div className="row">
            <NumField
              label="temperature (0-2)"
              step={0.1}
              min={0}
              max={2}
              value={settings.temperature}
              onSave={(v) => void saveDefaults({ temperature: v })}
            />
            <NumField
              label="max_tokens — output cap (0 = unbounded)"
              min={0}
              value={settings.max_tokens}
              onSave={(v) => void saveDefaults({ max_tokens: v })}
            />
            <NumField
              label="history_window — past turns (0 = full)"
              min={0}
              value={settings.history_window}
              onSave={(v) => void saveDefaults({ history_window: v })}
            />
          </div>
        </div>
      )}
    </>
  );
}
