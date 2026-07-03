// Typed client for the PumpkinSpice FastAPI backend, including SSE streaming
// for chat (POST) and live run events.

// The decoder endpoint PumpkinSpice talks to. Default localhost; the backend's
// PUMPKINSPICE_LMSTUDIO_URL is authoritative for runs/settings, this is the
// browser-side default for the badge/chat polls.
export const DECODER_URL = "http://localhost:1234";

export interface LoadedModel {
  id: string;
  arch?: string;
  quantization?: string;
  loaded_context_length?: number;
}

export interface Backend {
  name: string;
  base_url: string;
}
export interface ConfigInfo {
  name: string;
  task: string;
  retrieval?: string;
  world?: string;
  max_turns?: number;
}
export interface CaptureInfo {
  name: string;
  turns: number;
}
export interface Turn {
  index: number;
  task: string;
  world_state: Record<string, unknown>;
  retrieval: { backend?: string; latency_ms?: number; nodes?: { id: string; score: number; text: string }[] };
  prompt: string;
  raw_output: string;
  action: { kind: string; args: Record<string, unknown> };
  outcome: { ok: boolean; status_code: number; error: string | null };
  timings_ms: Record<string, number>;
  reasoning?: string;
  decoder_empty?: boolean;
  plan?: string; // committed plan (Stage 2 planning strategy)
}

export interface MapTile {
  x: number;
  y: number;
  name?: string;
  content: { type: string; code: string } | null;
}
export interface RunDetail {
  id: string;
  config_name: string;
  task: string;
  plugins: Record<string, string>;
  status: string;
  error: string | null;
  turns: Turn[];
}

// --- API auth token (bearer) ---
// Stored in localStorage and attached as `Authorization: Bearer <token>` on every
// /api call. Only needed when the server is started with PUMPKINSPICE_API_TOKEN;
// when unset the header is omitted and the server accepts requests as before.
const TOKEN_KEY = "ps_api_token";
export function getApiToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}
export function setApiToken(t: string): void {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}
function authHeaders(): Record<string, string> {
  const t = getApiToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}
const AUTH_ERR =
  "401: API token missing or invalid. The server requires PUMPKINSPICE_API_TOKEN auth; set the token in Settings > API token.";

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers as Record<string, string> | undefined) },
  });
  if (r.status === 401) throw new Error(AUTH_ERR);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return (await r.json()) as T;
}

const jsonPost = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getBackends = () => j<Backend[]>("/api/backends");
export const getModels = (baseUrl: string) =>
  j<string[]>(`/api/decoder/models?base_url=${encodeURIComponent(baseUrl)}`);
export const getConfigs = () => j<ConfigInfo[]>("/api/configs");
export const getCaptures = () => j<CaptureInfo[]>("/api/captures");
export const getWorldMap = (baseUrl: string) =>
  j<MapTile[]>(`/api/world/map?base_url=${encodeURIComponent(baseUrl)}`);
export const getLoadedModels = () =>
  j<LoadedModel[]>(`/api/decoder/loaded?base_url=${encodeURIComponent(DECODER_URL)}`);
export const getRetrievalOptions = () => j<string[]>("/api/retrieval-options");
export const getPromptOptions = () => j<string[]>("/api/prompt-options");

// --- Model under test + decode defaults (Phase 2) ---
export interface ModelSettings {
  model: string;
  temperature: number;
  max_tokens: number;
  history_window: number;
}
export const getModelSettings = () => j<ModelSettings>("/api/model/settings");
export const updateModelSettings = (body: Partial<ModelSettings>) =>
  j<ModelSettings>("/api/model/settings", jsonPost(body));
export const getAvailableModels = () => j<string[]>("/api/model/available");
export const loadModel = (model: string) =>
  j<{ loaded: string }>("/api/model/load", jsonPost({ model }));

// --- MCP servers (Phase 3a; Chat-only) ---
export interface McpServer {
  name: string;
  command: string;
  args: string[];
  enabled: boolean;
}
export const getMcpServers = () => j<McpServer[]>("/api/mcp/servers");
export const upsertMcpServer = (s: McpServer) => j<McpServer>("/api/mcp/servers", jsonPost(s));
export const setMcpEnabled = (name: string, enabled: boolean) =>
  j<{ enabled: boolean }>(
    `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
    jsonPost({ enabled }),
  );
export const deleteMcpServer = (name: string) =>
  j<{ deleted: boolean }>(`/api/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
export const startRunOptions = (
  retrieval: string,
  task: string,
  prompt: string,
  maxTurns: number,
  goalItem?: string,
  seed?: number,
  temperature?: number,
) =>
  j<{ id: string }>(
    "/api/runs",
    jsonPost({
      retrieval,
      task,
      prompt,
      max_turns: maxTurns,
      goal_item: goalItem || null,
      seed: seed ?? null,
      temperature: temperature ?? 0.7,
    }),
  );
export const stopRun = (id: string) =>
  j<{ stopping: boolean }>(`/api/runs/${id}/stop`, { method: "POST" });
export const startTrials = (
  retrieval: string,
  task: string,
  prompt: string,
  maxTurns: number,
  goalItem: string,
  trials: number,
  temperature: number,
) =>
  j<{ batch: string; trials: number }>(
    "/api/trials",
    jsonPost({
      retrieval,
      task,
      prompt,
      max_turns: maxTurns,
      goal_item: goalItem || null,
      trials,
      temperature,
    }),
  );
// --- Hanoi: a second, vocabulary-disjoint benchmark playground ---
export const startHanoiRun = (
  prompt: string,
  maxTurns: number,
  disks: number,
  seed?: number,
  temperature?: number,
) =>
  j<{ id: string }>(
    "/api/hanoi/runs",
    jsonPost({
      prompt,
      max_turns: maxTurns,
      disks,
      seed: seed ?? null,
      temperature: temperature ?? 0.7,
    }),
  );
export const startHanoiTrials = (
  prompt: string,
  maxTurns: number,
  disks: number,
  trials: number,
  temperature: number,
) =>
  j<{ batch: string; trials: number }>(
    "/api/hanoi/trials",
    jsonPost({ prompt, max_turns: maxTurns, disks, trials, temperature }),
  );

export const getCapture = (name: string) => j<Turn[]>(`/api/captures/${encodeURIComponent(name)}`);
export const startRun = (config: string) => j<{ id: string }>("/api/runs", jsonPost({ config }));
export const getRun = (id: string) => j<RunDetail>(`/api/runs/${id}`);

export interface RunSummary {
  id: string;
  config_name: string;
  status: string;
  turns: number;
  task: string;
  tags: string[];
}
export const getRuns = () => j<RunSummary[]>("/api/runs");
export const stopBatch = (batchId: string) =>
  j<{ stopping: boolean }>(`/api/trials/${encodeURIComponent(batchId)}/stop`, { method: "POST" });

// --- Reports (persistent run registry) ---
export interface RunMetrics {
  steps?: number;
  success?: boolean | null;
  failed_actions?: number;
  revisits?: number;
  replans?: number;
  decode_tok_s?: number;
  avg_decode_ms?: number;
  [k: string]: unknown;
}
export interface RunReport {
  id: string;
  benchmark: string;
  model: string;
  strategy: string;
  retrieval: string;
  task: string;
  goal: string;
  status: string;
  finished_at: string;
  metrics: RunMetrics;
  label: string;
  tags: string[];
  notes: string;
  capture_path: string;
}
export interface LeaderboardEntry {
  model: string;
  runs: number;
  successes: number;
  success_rate: number;
  best_steps: number | null;
  avg_steps: number | null;
}
export const getReportRuns = (q: Record<string, string> = {}) =>
  j<RunReport[]>("/api/reports/runs?" + new URLSearchParams(q).toString());
export const getLeaderboard = (benchmark = "herobench") =>
  j<LeaderboardEntry[]>(`/api/reports/leaderboard?benchmark=${benchmark}`);
export const updateReportRun = (
  id: string,
  body: { label?: string; tags?: string[]; notes?: string },
) => j<RunReport>(`/api/reports/runs/${id}`, jsonPost(body));

export interface ChatRequest {
  base_url: string;
  model?: string;
  messages: { role: string; content: string }[];
  max_tokens?: number;
  sampler?: Record<string, unknown>;
}

export interface ChatChunk {
  delta?: string; // final-answer content
  reasoning?: string; // chain-of-thought (reasoning models)
  tool_call?: { name: string; args: Record<string, unknown> }; // MCP tool invoked
  tool_result?: { name: string; result: string }; // its result
}

export async function chatStream(req: ChatRequest, onChunk: (c: ChatChunk) => void): Promise<void> {
  const init = jsonPost(req);
  const r = await fetch("/api/chat", {
    ...init,
    headers: { ...(init.headers as Record<string, string>), ...authHeaders() },
  });
  await consumeSSE(r, (obj) => {
    if (typeof obj.delta === "string") onChunk({ delta: obj.delta });
    if (typeof obj.reasoning === "string") onChunk({ reasoning: obj.reasoning });
    if (obj.tool_call) onChunk({ tool_call: obj.tool_call as ChatChunk["tool_call"] });
    if (obj.tool_result) onChunk({ tool_result: obj.tool_result as ChatChunk["tool_result"] });
    if (obj.error) throw new Error(String(obj.error) + (obj.detail ? `: ${obj.detail}` : ""));
  });
}

export async function runEventStream(
  id: string,
  onTurn: (t: Turn) => void,
  onEnd: (status: string, error: string | null) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const r = await fetch(`/api/runs/${id}/events`, { headers: authHeaders(), signal });
    await consumeSSE(r, (obj) => {
      if (obj.event === "turn") onTurn(obj.turn as Turn);
      else if (obj.event === "end") onEnd(String(obj.status), obj.error as string | null);
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return; // superseded watch
    throw e;
  }
}

type SSEObject = Record<string, unknown>;

async function consumeSSE(r: Response, onData: (obj: SSEObject) => void): Promise<void> {
  if (r.status === 401) throw new Error(AUTH_ERR);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  if (!r.body) throw new Error("no response stream");
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data) continue;
        let obj: SSEObject;
        try {
          obj = JSON.parse(data) as SSEObject;
        } catch {
          continue; /* ignore malformed frame */
        }
        // dispatch OUTSIDE the parse guard: onData legitimately throws to
        // propagate server error frames (e.g. chat {error} events)
        onData(obj);
      }
    }
  }
}
