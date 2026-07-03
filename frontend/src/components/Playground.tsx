import { useEffect, useRef, useState } from "react";
import { DECODER_URL, chatStream } from "../api";

interface ToolUse {
  name: string;
  args: Record<string, unknown>;
  result?: string;
}
interface Msg {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  tools?: ToolUse[];
}

export function Playground() {
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [msgs]);

  async function send() {
    const content = input.trim();
    if (!content || busy) return;
    setErr("");
    setInput("");
    const history = [...msgs, { role: "user" as const, content }];
    setMsgs([...history, { role: "assistant", content: "", reasoning: "" }]);
    setBusy(true);
    try {
      // model is ambient: omitted, so LMStudio uses whatever it has loaded
      await chatStream({ base_url: DECODER_URL, messages: history, max_tokens: 0 }, (c) => {
        setMsgs((m) => {
          const copy = m.slice();
          const last = { ...copy[copy.length - 1] };
          if (c.delta) last.content += c.delta;
          if (c.reasoning) last.reasoning = (last.reasoning ?? "") + c.reasoning;
          if (c.tool_call) last.tools = [...(last.tools ?? []), { ...c.tool_call }];
          if (c.tool_result) {
            const tools = (last.tools ?? []).slice();
            // attach the result to the most recent pending call
            for (let k = tools.length - 1; k >= 0; k--) {
              if (tools[k].result === undefined) {
                tools[k] = { ...tools[k], result: c.tool_result.result };
                break;
              }
            }
            last.tools = tools;
          }
          copy[copy.length - 1] = last;
          return copy;
        });
      });
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="card chat">
        {msgs.length === 0 && (
          <div className="muted">
            Chat with the loaded model (shown in the header). Reasoning models stream their thinking
            in a muted block, then the answer.
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.reasoning && <div className="thinking">💭 {m.reasoning}</div>}
            {m.tools?.map((t, k) => (
              <div key={k} className="toolcall">
                🔧 <strong>{t.name}</strong>
                <span className="mono"> {JSON.stringify(t.args)}</span>
                {t.result !== undefined ? (
                  <div className="toolresult mono">&rarr; {t.result}</div>
                ) : (
                  <span className="muted"> …</span>
                )}
              </div>
            ))}
            {m.content ||
              (busy && i === msgs.length - 1 && !m.reasoning && !m.tools?.length ? "…" : "")}
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <div className="card">
        {err && <div className="err" style={{ marginBottom: 8 }}>{err}</div>}
        <div className="row">
          <textarea
            style={{ flex: 1, minHeight: 52 }}
            placeholder="Message the loaded model…  (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <button className="btn" disabled={busy} onClick={() => void send()}>
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </>
  );
}
