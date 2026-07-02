import { useState, useRef, useEffect, useMemo, useCallback, memo } from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

/*
 * AgentFlow chat interface.
 *
 * Design reference: UI_DESIGN.md
 *
 * Phase 8b: live SSE streaming from FastAPI backend, review interrupt
 * resolution, and PDF upload. The thread_id is generated once on mount
 * via crypto.randomUUID() and persists for the lifetime of the page.
 *
 * API base: VITE_API_BASE is injected by vite.config.js (default "" —
 * relative paths go through Vite's dev/preview proxy to :8000). Set
 * VITE_API_BASE=https://my-deploy.example.com to bypass the proxy and
 * call the API directly when serving the static bundle from a
 * different host.
 */

const API_BASE = import.meta.env.VITE_API_BASE || "";
const apiUrl = (path) => `${API_BASE}${path}`;

function apiFetch(path, options = {}) {
  return fetch(apiUrl(path), options);
}

import { MAX_MESSAGE_CHARS, MAX_UPLOAD_BYTES, TRACE_STREAM_NODES } from "./constants";
import { parseSSEPayload } from "./sseParser";

function streamAgentMeta(agent) {
  if (agent === "chat_agent") return "chat";
  if (agent === "research_agent") return "research";
  if (agent === "analysis_agent") return "analysis";
  return agent;
}

function agentLabelFromRoute(route) {
  if (route === "chat") return "chat_agent";
  if (route === "research") return "research_agent";
  if (route === "analysis") return "analysis_agent";
  return "synthesizer";
}

function formatLastSeen(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : "—";
}

const AGENT_COLORS = {
  router: "var(--af-router)",
  research_agent: "var(--af-research)",
  analysis_agent: "var(--af-analysis)",
  chat_agent: "var(--af-chat)",
  synthesizer: "var(--af-synthesizer)",
  human_review: "var(--af-review)",
};

// Citation parser: the synthesizer emits "Sources: [1] https://… [2] https://…"
// in `final_response`. Extract (n, url) pairs and a friendly host label so
// the UI can render clickable chips. Host strips a leading "www." for visual
// cleanliness (chips for "en.wikipedia.org" not "www.en.wikipedia.org").
const uuid = () => {
  const webCrypto = globalThis.crypto;
  if (webCrypto?.randomUUID) return webCrypto.randomUUID();
  if (!webCrypto?.getRandomValues) {
    throw new Error("Web Crypto API is required to generate thread IDs");
  }
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
    (c ^ (webCrypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16)
  );
};

function parseCitations(text) {
  if (!text) return [];
  const out = [];
  let m;
  const re = /\[(\d+)\]\s+(https?:\/\/\S+)/g;
  try {
    while ((m = re.exec(text)) !== null) {
      let host = m[2];
      try {
        host = new URL(m[2]).hostname.replace(/^www\./, "");
      } catch {
        // Malformed URL — fall back to the raw match
      }
      out.push({ n: Number(m[1]), url: m[2], host });
    }
  } catch {
    return out;
  }
  return out;
}

function now() {
  return new Date().toLocaleTimeString("en-GB", { hour12: false });
}

const TRACE_RAIL_STYLE = {
  width: 176,
  flexShrink: 0,
  background: "var(--af-bg-panel)",
  borderRight: "1px solid var(--af-border)",
  padding: "14px 12px",
  overflowY: "auto",
};

const INPUT_STYLE_BASE = {
  flex: 1,
  background: "var(--af-bg-panel)",
  border: "1px solid var(--af-border)",
  borderRadius: 6,
  padding: "8px 12px",
  color: "var(--af-text-primary)",
  fontFamily: "var(--af-font-sans)",
  fontSize: 13,
  outline: "none",
};

const SEND_BUTTON_BASE_STYLE = {
  background: "transparent",
  border: "none",
  color: "var(--af-synthesizer)",
  fontSize: 18,
  lineHeight: 1,
  padding: 6,
};

function TraceRail({ trace }) {
  return (
    <div
      className="af-trace-rail af-scroll"
      style={TRACE_RAIL_STYLE}
    >
      <div
        style={{
          fontFamily: "var(--af-font-mono)",
          fontSize: 10.5,
          letterSpacing: "0.06em",
          color: "var(--af-text-muted)",
          marginBottom: 14,
        }}
      >
        EXECUTION TRACE
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
        {trace.length === 0 ? (
          <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10, color: "var(--af-text-muted)", fontStyle: "italic", opacity: 0.6 }}>
            Waiting for first request...
          </div>
        ) : trace.map((entry, i) => (
          <div key={i} style={{ display: "flex", gap: 8 }}>
            <div
              className={entry.active ? "af-pulse" : ""}
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: AGENT_COLORS[entry.node] || "var(--af-router)",
                marginTop: 4,
                flexShrink: 0,
              }}
            />
            <div>
              <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 11, color: "var(--af-text-primary)", display: "flex", alignItems: "center", gap: 4 }}>
                <span>
                  {entry.node === "router" ? "🚦" :
                   entry.node === "research_agent" ? "🔍" :
                   entry.node === "analysis_agent" ? "📊" :
                   entry.node === "chat_agent" ? "💬" :
                   entry.node === "synthesizer" ? "✍️" :
                   entry.node === "human_review" ? "🛑" : "🤖"}
                </span>
                {entry.node}
              </div>
              <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10, color: "var(--af-text-muted)" }}>
                {entry.label}{entry.latency ? ` · ${entry.latency}` : ""} · {entry.time}
                {entry.active && <span className="af-trace-spinner" style={{ marginLeft: 6 }}>↻</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AgentAvatar({ agent }) {
  const icon = agent === "router" ? "🚦" :
               agent === "research_agent" ? "🔍" :
               agent === "analysis_agent" ? "📊" :
               agent === "chat_agent" ? "💬" :
               agent === "synthesizer" ? "✍️" :
               agent === "human_review" ? "🛑" : "🤖";
  const bg = AGENT_COLORS[agent] || "var(--af-router)";
  return (
    <div style={{
      width: 20, height: 20, borderRadius: "50%", background: bg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "var(--af-bg-panel)", flexShrink: 0
    }}>
      {icon}
    </div>
  );
}

function CopyButton({ text, style }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } catch {
          setCopied(false);
        }
      }}
      title="Copy"
      style={{
        background: "transparent",
        border: "none",
        color: copied ? "var(--af-research)" : "var(--af-text-muted)",
        cursor: "pointer",
        padding: "2px 4px",
        fontFamily: "var(--af-font-sans)",
        fontSize: 12,
        ...style
      }}
    >
      {copied ? "✓ Copied" : "📋"}
    </button>
  );
}

const COLLAPSE_THRESHOLD = 1500; // chars

const Message = memo(function Message({ msg, onApprove, onEditResend, onSubmitEdit, editingReview, editText, setEditText, onRetry, onInlineEdit }) {
  const [collapsed, setCollapsed] = useState(
    msg.role === "agent" && !msg.streaming && (msg.text?.length ?? 0) > COLLAPSE_THRESHOLD
  );
  const [isTyping, setIsTyping] = useState(false);
  const [timeLeft, setTimeLeft] = useState(60);

  useEffect(() => {
    if (!msg.streaming) return;
    setIsTyping(true);
    const t = setTimeout(() => setIsTyping(false), 300);
    return () => clearTimeout(t);
  }, [msg.text, msg.streaming]);

  useEffect(() => {
    if (msg.role !== "review") return;
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [msg.role]);

  // Auto-uncollapse once streaming finishes
  useEffect(() => {
    if (!msg.streaming && (msg.text?.length ?? 0) <= COLLAPSE_THRESHOLD) {
      setCollapsed(false);
    }
  }, [msg.streaming, msg.text]);

  if (msg.role === "user") {
    return (
      <div
        style={{
          alignSelf: "flex-end",
          maxWidth: "78%",
          background: "var(--af-bg-surface)",
          borderRadius: 8,
          padding: "9px 13px",
          color: "var(--af-text-primary)",
          fontSize: 13.5,
          lineHeight: 1.5,
          whiteSpace: "pre-wrap",
          position: "relative",
        }}
        className="af-user-message af-msg-enter"
      >
        <div style={{ position: "relative" }}>
          {msg.text}
          {onInlineEdit && (
            <button
              className="af-inline-edit-btn"
              onClick={() => onInlineEdit(msg.text)}
              title="Edit text in input box"
            >
              ✏️
            </button>
          )}
        </div>
        {msg.timestamp && (
          <div className="af-msg-timestamp" style={{ fontSize: 10, color: "var(--af-text-muted)", marginTop: 4, textAlign: "right", fontFamily: "var(--af-font-mono)" }}>
            {msg.timestamp}
          </div>
        )}
      </div>
    );
  }

  if (msg.role === "review") {
    if (editingReview) {
      return (
        <div
          style={{
            maxWidth: "92%",
            background: "var(--af-review-bg)",
            border: "1px solid var(--af-review-border)",
            borderRadius: 8,
            padding: "12px 14px",
          }}
        >
          <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-review)", marginBottom: 7 }}>
            HUMAN_REVIEW · editing
          </div>
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--af-review)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--af-review-border)";
            }}
            rows={4}
            style={{
              width: "100%",
              background: "var(--af-bg-app)",
              border: "1px solid var(--af-review-border)",
              borderRadius: 5,
              color: "var(--af-text-body)",
              fontFamily: "var(--af-font-sans)",
              fontSize: 13,
              padding: 8,
              outline: "none",
              resize: "vertical",
              marginBottom: 8,
            }}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={onSubmitEdit}
              style={{
                fontFamily: "var(--af-font-mono)",
                fontSize: 11.5,
                color: "var(--af-bg-app)",
                background: "var(--af-review)",
                border: "none",
                borderRadius: 5,
                padding: "6px 12px",
                cursor: "pointer",
              }}
            >
              Submit
            </button>
          </div>
        </div>
      );
    }
    return (
      <div
        style={{
          maxWidth: "92%",
          background: "var(--af-review-bg)",
          border: "1px solid var(--af-review-border)",
          borderRadius: 8,
          padding: "12px 14px",
        }}
      >
        <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-review)", marginBottom: 7, display: "flex", justifyContent: "space-between" }}>
          <span>HUMAN_REVIEW · awaiting approval</span>
          <span style={{ opacity: 0.7 }}>{timeLeft}s</span>
        </div>
        <div style={{ color: "var(--af-text-body)", fontSize: 13.5, lineHeight: 1.6, marginBottom: 11 }}>
          {msg.text}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={onApprove}
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 11.5,
              color: "var(--af-bg-app)",
              background: "var(--af-review)",
              border: "none",
              borderRadius: 5,
              padding: "6px 12px",
              cursor: "pointer",
            }}
          >
            Approve
          </button>
          <button
            onClick={() => onEditResend(msg.text)}
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 11.5,
              color: "var(--af-text-body)",
              background: "transparent",
              border: "1px solid var(--af-review-border)",
              borderRadius: 5,
              padding: "6px 12px",
              cursor: "pointer",
            }}
          >
            Edit and resend
          </button>
        </div>
      </div>
    );
  }

  // role === "agent"
  const color = (msg.error || msg.aborted) ? "var(--af-error)" : (AGENT_COLORS[msg.agent] || "var(--af-router)");
  // Phase 2: citation chips. Parsed once per text change via useMemo in
  // the parent (App) and passed in as `citations` to keep this memoized
  // component pure. Hidden while streaming so chips don't pop in mid-
  // stream before the LLM has finished writing.
  const citations = msg.citations ?? [];
  return (
    <div className="af-msg-enter" style={{ maxWidth: "84%", borderLeft: `2px solid ${color}`, padding: "6px 0 6px 12px" }}>
      <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color, marginBottom: 5, display: "flex", alignItems: "center", gap: 8 }}>
        <AgentAvatar agent={msg.agent} />
        <span>
          <span style={{ fontWeight: 500, letterSpacing: "0.04em" }}>{msg.agent ? msg.agent.toUpperCase().replace("_AGENT", "") : "AGENT"}</span> · {msg.error ? "error" : msg.aborted ? "aborted" : msg.meta}
        </span>
        {msg.text && !msg.streaming && !msg.error && !msg.aborted && (
          <CopyButton text={msg.text} />
        )}
        {msg.timestamp && (
          <span className="af-msg-timestamp" style={{ color: "var(--af-text-muted)", marginLeft: "auto" }}>{msg.timestamp}</span>
        )}
      </div>
      <div style={{ color: "var(--af-text-body)", fontSize: 13.5, lineHeight: 1.6 }}>
        {msg.error ? (
          <div className="af-error-panel" role="alert">
            <div className="af-error-panel-text">{msg.text || "An error occurred."}</div>
            {onRetry ? (
              <button
                className="af-error-panel-retry"
                onClick={() => onRetry(msg.id)}
              >
                Retry
              </button>
            ) : null}
          </div>
        ) : msg.aborted ? (
          <div className="af-aborted">[aborted] — generation was cancelled.</div>
        ) : (
          <>
            {msg.text ? (
              <>
                <div className={`markdown-body${collapsed ? " af-msg-collapsed" : ""}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || "");
                        const lang = match ? match[1] : null;
                        const codeText = String(children).replace(/\n$/, "");
                        return !inline && match ? (
                          <div className="af-code-block">
                            <div className="af-code-header">
                              <span>{lang}</span>
                              <CopyButton text={codeText} />
                            </div>
                            <SyntaxHighlighter
                              style={vscDarkPlus}
                              language={lang}
                              PreTag="div"
                              customStyle={{ margin: 0, padding: "12px", background: "var(--af-bg-panel)", fontSize: "12px" }}
                            >
                              {codeText}
                            </SyntaxHighlighter>
                          </div>
                        ) : (
                          <code className={className} {...props}>{children}</code>
                        );
                      }
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
                {(msg.text?.length ?? 0) > COLLAPSE_THRESHOLD && !msg.streaming && (
                  <button
                    className="af-msg-expand-btn"
                    onClick={() => setCollapsed((c) => !c)}
                  >
                    {collapsed ? `▼ Show full response (${msg.text.length} chars)` : "▲ Collapse"}
                  </button>
                )}
              </>
            ) : (msg.streaming && (
              <div className="af-typing-indicator">
                <span />
                <span />
                <span />
              </div>
            ))}
            {msg.streaming ? <span className={`af-cursor ${isTyping ? "af-cursor-typing" : ""}`} aria-hidden="true" /> : null}
          </>
        )}
      </div>
      {!msg.error && !msg.aborted && citations.length > 0 ? (
        <div
          style={{
            marginTop: 8,
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
            alignItems: "center",
          }}
        >
          <span
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 10,
              color: "var(--af-text-muted)",
            }}
          >
            Sources:
          </span>
          {citations.slice(0, 8).map((c, i) => (
            <a
              key={`${c.n}-${i}`}
              className="af-source-chip"
              href={c.url}
              target="_blank"
              rel="noreferrer noopener"
            >
              [{c.n}] {c.host}
            </a>
          ))}
          {citations.length > 8 ? (
            <span
              style={{
                fontFamily: "var(--af-font-mono)",
                fontSize: 10,
                color: "var(--af-text-muted)",
              }}
            >
              …
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
});

export default function App() {
  const [threadId, setThreadId] = useState(() => uuid());
  const [messages, setMessages] = useState([]);
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);
  const [trace, setTrace] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [reviewRequired, setReviewRequired] = useState(false);
  const [editingReview, setEditingReview] = useState(false);
  const [editText, setEditText] = useState("");
  const [statusError, setStatusError] = useState(null);
  const [showScrollChip, setShowScrollChip] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const pendingDraftRef = useRef(null);
  const rafPendingRef = useRef(false);
  const shortcutsButtonRef = useRef(null);
  const shortcutsPopoverRef = useRef(null);
  const autoGrowRafRef = useRef(null);
  // Per-stream generation counter. Incremented at the start of every
  // sendMessage call; every async callback captures its own snapshot and
  // bails out if the ref has moved on (i.e. a newer request has started
  // or resetThread was called). This prevents a stale fetch/watchdog/rAF
  // from mutating state that now belongs to a different conversation.
  const streamGenRef = useRef(0);
  const loadGenRef = useRef(0);
  const activeStreamAgentRef = useRef("router");
  // Phase 2: stream timing. `synthStartMs` is captured the moment the
  // streaming message appears in the UI; `lastMetaUpdateMs` gates the
  // rAF's meta write to 4 Hz so a 60Hz rAF doesn't re-render the
  // message list on every frame just to update a single label.
  const synthStartMsRef = useRef(null);
  const lastMetaUpdateMsRef = useRef(0);
  // Per-stream source count. Backend emits [SOURCES:n] right before
  // [DONE] only on a non-interrupt run; the meta line picks it up.
  const streamSourcesRef = useRef(0);
  // Watchdog: timestamp of the last received token. Reset to Date.now()
  // on every successful read. Interval checks every 5s and triggers an
  // error state if the gap exceeds STALL_MS (60s).
  const lastTokenAtMsRef = useRef(0);

  const lastMessagesLengthRef = useRef(0);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!showScrollChip) {
      el.scrollTo({ top: el.scrollHeight });
      setUnreadCount(0);
    } else {
      if (messages.length > lastMessagesLengthRef.current) {
        setUnreadCount(prev => prev + (messages.length - lastMessagesLengthRef.current));
      }
    }
    lastMessagesLengthRef.current = messages.length;
  }, [messages, showScrollChip]);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    setShowScrollChip(!isAtBottom);
  }

  const messagesWithCitations = useMemo(() => {
    return messages.map((msg, i) => {
      if (msg.role !== "agent" || msg.agent !== "synthesizer" || msg.error || msg.streaming) return msg;
      const citations = parseCitations(msg.text);
      return citations.length ? { ...msg, citations } : msg;
    });
  }, [messages]);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  // Ref that always reflects the latest isStreaming value inside the
  // keydown event listener (which captures a stale closure over the
  // initial false). Updated synchronously after every isStreaming change.
  const isStreamingRef = useRef(isStreaming);
  useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);

  // Shared reset: abort any in-flight stream first so the running fetch
  // reader stops updating state, then invalidate the current generation,
  // mint a new threadId, and clear everything. Called by both the New
  // button and the Cmd/Ctrl+K shortcut.
  const resetThread = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    streamGenRef.current += 1; // invalidate any in-flight callbacks
    setThreadId(uuid());
    setMessages([]);
    setTrace([]);
    setInput("");
    setIsStreaming(false);
    setShowScrollChip(false);
    setReviewRequired(false);
    setEditingReview(false);
    setEditText("");
    setStatusError(null);
  }, []);

  // Esc key handling also closes the shortcuts popover. The popover's
  // own Esc handler can't run because focus is outside the popover when
  // it's open; this top-level handler works in both cases.
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === "Escape" && showShortcuts) {
        setShowShortcuts(false);
        shortcutsButtonRef.current?.focus();
        return;
      }
      if (e.key === "Escape" && isStreamingRef.current && abortRef.current) {
        abortRef.current.abort();
        setMessages((m) => {
          if (m.length === 0) return m;
          const next = [...m];
          const last = next[next.length - 1];
          if (last.role === "agent" && last.streaming) {
            next[next.length - 1] = {
              ...last,
              streaming: false,
              aborted: true,
              meta: "aborted",
            };
          }
          return next;
        });
        setIsStreaming(false);
        setTrace((t) => [
          ...t.map((x) => (x.active ? { ...x, active: false } : x)),
          { node: "router", label: "aborted", time: now() },
        ]);
      } else if (e.key === "Escape" && !isStreamingRef.current) {
        setMessages((m) => {
          if (m.length === 0) return m;
          const next = [...m];
          const last = next[next.length - 1];
          if (last.role === "review") {
            next[next.length - 1] = {
              ...last,
              role: "agent",
              aborted: true,
              meta: "aborted",
            };
            setTrace((t) => [
              ...t.map((x) => (x.active ? { ...x, active: false } : x)),
              { node: "human_review", label: "aborted", time: now() },
            ]);
            setEditingReview(false);
          }
          return next;
        });
      }
      const tag = document.activeElement?.tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        const tag = document.activeElement?.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        e.preventDefault();
        resetThread();
      }
      if (e.key === "h" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setShowHistory((v) => !v);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [resetThread, showShortcuts]);

  // Outside-click closes the shortcuts popover. Listener is bound only
  // while the popover is open; the popover ref + button ref let us
  // ignore clicks on either (those re-open or are no-ops).
  useEffect(() => {
    if (!showShortcuts) return;
    function handleMouseDown(e) {
      const popover = shortcutsPopoverRef.current;
      const btn = shortcutsButtonRef.current;
      if (popover && popover.contains(e.target)) return;
      if (btn && btn.contains(e.target)) return;
      setShowShortcuts(false);
    }
    document.addEventListener("mousedown", handleMouseDown);
    return () => document.removeEventListener("mousedown", handleMouseDown);
  }, [showShortcuts]);

  function handleRetry(msgId) {
    const idx = messages.findIndex((m) => m.id === msgId);
    if (idx === -1) return;
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        setInput("");
        sendMessage(messages[i].text);
        return;
      }
    }
  }

  // Phase 2: extracted from handleSend so welcome-chip clicks can pass an
  // explicit text without going through `input` state (which would require
  // a setTimeout to commit before sending). `input` argument is REQUIRED
  // — pass `input.trim()` from the input bar handler.
  async function sendMessage(text) {
    if (!text || !text.trim() || isStreaming) return;
    if (text.length > MAX_MESSAGE_CHARS) {
      setStatusError(`message too long (max ${MAX_MESSAGE_CHARS} characters)`);
      setTimeout(() => setStatusError(null), 3000);
      return;
    }
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    // Mint a new generation token. Every async callback below captures
    // `myGen` and skips state writes if `streamGenRef.current !== myGen`.
    const myGen = streamGenRef.current + 1;
    streamGenRef.current = myGen;
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages((m) => [...m, { role: "user", text, id: uuid(), timestamp }]);
    setTrace((t) => [...t, { node: "router", label: "routing…", active: true, time: now() }]);
    setIsStreaming(true);
    // Reset stream-time refs at the start of each request.
    synthStartMsRef.current = null;
    lastMetaUpdateMsRef.current = 0;
    streamSourcesRef.current = 0;
    activeStreamAgentRef.current = "router";

    let res;
    try {
      res = await apiFetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadId,
          message: text,
          review_required: reviewRequired,
        }),
        signal: abortRef.current.signal,
      });
    } catch (err) {
      if (err.name === "AbortError") {
        setIsStreaming(false);
        // Pre-first-token abort: the streaming message was never created.
        // Surface a stub so the user sees feedback instead of a frozen UI.
        setMessages((m) => {
          if (m.length === 0) return m;
          const next = [...m];
          const last = next[next.length - 1];
          if (last.role === "agent" && last.streaming) {
            next[next.length - 1] = {
              ...last,
              streaming: false,
              aborted: true,
              meta: "aborted",
            };
          } else if (last.role === "user") {
            next.push({
              role: "agent",
              agent: "router",
              meta: "aborted",
              text: "[aborted before first token]",
              aborted: true,
              id: uuid(),
            });
          }
          return next;
        });
        setTrace((t) => [
          ...t.map((x) => (x.active ? { ...x, active: false } : x)),
          { node: "router", label: "aborted", time: now() },
        ]);
        return;
      }
      setIsStreaming(false);
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: "router", label: "error", time: now() },
      ]);
      setStatusError(err.message || "network error");
      setTimeout(() => setStatusError(null), 3000);
      return;
    }

    // Non-2xx response — the body isn't a valid SSE stream. Surface the
    // status and bail out cleanly. The 400 case here is the backend's
    // thread_id regex rejection.
    if (!res.ok) {
      setIsStreaming(false);
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: "router", label: `error ${res.status}`, time: now() },
      ]);
      let detail = `server returned ${res.status}`;
      try {
        const errBody = await res.json();
        if (errBody?.detail) {
          detail = typeof errBody.detail === "string"
            ? errBody.detail
            : JSON.stringify(errBody.detail);
        }
      } catch { /* non-JSON body */ }
      setStatusError(detail);
      setTimeout(() => setStatusError(null), 3000);
      return;
    }

    const streamingId = uuid();
    const agentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages((m) => [
      ...m,
      {
        role: "agent",
        agent: "router",
        meta: "routing…",
        text: "",
        streaming: true,
        id: streamingId,
        timestamp: agentTimestamp,
      },
    ]);
    synthStartMsRef.current = Date.now();
    lastMetaUpdateMsRef.current = Date.now();
    lastTokenAtMsRef.current = Date.now();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let draft = "";
    let sentinel = null;
    let sourcesCount = 0;

    // Watchdog: if no token arrives for 60s, surface as [ERROR] so the
    // user isn't staring at a frozen cursor. The setInterval is cleared
    // on every successful read and at the end of the stream.
    const STALL_MS = 60_000;
    const watchdog = setInterval(() => {
      if (streamGenRef.current !== myGen) { clearInterval(watchdog); return; }
      if (Date.now() - lastTokenAtMsRef.current > STALL_MS) {
        sentinel = "[ERROR]";
        clearInterval(watchdog);
        try { reader.cancel(); } catch { /* best effort */ }
        if (streamGenRef.current !== myGen) return;
        setMessages((m) => {
          const next = [...m];
          const idx = next.findIndex((x) => x.id === streamingId);
          if (idx !== -1 && next[idx].streaming) {
            next[idx] = {
              ...next[idx],
              streaming: false,
              error: true,
              text: next[idx].text || "Stream stalled (60s with no tokens).",
            };
          }
          return next;
        });
        setTrace((t) => [
          ...t.map((e) => (e.active ? { ...e, active: false } : e)),
          { node: "synthesizer", label: "stalled", time: now() },
        ]);
        setIsStreaming(false);
      }
    }, 5_000);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        lastTokenAtMsRef.current = Date.now();
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const rawPayload = line.slice(6);
          const parsed = parseSSEPayload(rawPayload);

          if (parsed.kind === "done") {
            sentinel = "[DONE]";
            break;
          }
          if (parsed.kind === "interrupt") {
            sentinel = "[INTERRUPT]";
            break;
          }
          if (parsed.kind === "error") {
            sentinel = "[ERROR]";
            break;
          }
          if (parsed.kind === "sources") {
            sourcesCount = parsed.value;
            streamSourcesRef.current = parsed.value;
            continue;
          }
          if (parsed.kind === "final") {
            if (!draft) {
              draft = parsed.value;
              pendingDraftRef.current = draft;
            }
            continue;
          }
          if (parsed.kind === "tool_start") {
            setTrace((t) => [
              ...t.map((e) => (e.active ? { ...e, active: false } : e)),
              { node: parsed.value, label: "tool…", active: true, time: now() },
            ]);
            continue;
          }
          if (parsed.kind === "node_start") {
            const { node, startMs } = parsed.value;
            setTrace((t) => {
              if (t.length > 0 && t[t.length - 1].node === node) return t;
              const next = t.map((e) => (e.active ? { ...e, active: false } : e));
              next.push({ node, label: "working…", active: true, time: now(), startMs });
              return next;
            });
            if (TRACE_STREAM_NODES.has(node) && node !== "router") {
              activeStreamAgentRef.current = node;
              setMessages((m) => {
                const next = [...m];
                const idx = next.findIndex((x) => x.id === streamingId);
                if (idx === -1 || !next[idx].streaming) return next;
                const short = streamAgentMeta(node);
                next[idx] = {
                  ...next[idx],
                  agent: node,
                  meta: `${short} · working…`,
                };
                return next;
              });
            }
            continue;
          }
          if (parsed.kind === "node_end") {
            const node = parsed.value;
            setTrace((t) => {
              for (let i = t.length - 1; i >= 0; i--) {
                if (t[i].node === node && t[i].startMs) {
                  const ms = Date.now() - t[i].startMs;
                  const next = [...t];
                  next[i] = {
                    ...t[i],
                    label: "done",
                    active: false,
                    latency: (ms / 1000).toFixed(1) + "s",
                    startMs: null,
                  };
                  return next;
                }
              }
              return t;
            });
            continue;
          }
          if (parsed.kind === "skip") continue;

          draft += parsed.value;
          pendingDraftRef.current = draft;
          if (!rafPendingRef.current) {
            rafPendingRef.current = true;
            requestAnimationFrame(() => {
              rafPendingRef.current = false;
              if (streamGenRef.current !== myGen) return; // stale request
              const d = pendingDraftRef.current;
              // Phase 2: throttle the meta write to ~4Hz. The text write
              // runs every rAF tick so tokens stay smooth; the meta line
              // is `synthesizer · N.Ns` and a 4Hz cadence is the right
              // resolution. Last-write-wins means a slow update is
              // harmless — the next frame will catch up.
              const nowMs = Date.now();
              const metaDue = nowMs - lastMetaUpdateMsRef.current >= 250;
              const startMs = synthStartMsRef.current ?? nowMs;
              const elapsed = ((nowMs - startMs) / 1000).toFixed(1) + "s";
              const short = streamAgentMeta(activeStreamAgentRef.current);
              setMessages((m) => {
                const next = [...m];
                // Find by streamingId so we never accidentally overwrite
                // the user message when React batches the initial add.
                const idx = next.findIndex((x) => x.id === streamingId);
                if (idx === -1) return next;
                next[idx] = {
                  ...next[idx],
                  text: d,
                  ...(metaDue ? { meta: `${short} · ${elapsed}` } : {}),
                };
                return next;
              });
              if (metaDue) lastMetaUpdateMsRef.current = nowMs;
            });
          }
        }
        if (sentinel) break;
      }
    } catch (err) {
      if (err.name === "AbortError") {
        if (streamGenRef.current === myGen) setIsStreaming(false);
        return;
      }
      if (streamGenRef.current !== myGen) return;
      setIsStreaming(false);
      setStatusError(err.message || "stream error");
      setTimeout(() => setStatusError(null), 3000);
      return;
    } finally {
      clearInterval(watchdog);
    }

    if (streamGenRef.current !== myGen) return; // superseded by a newer request
    setIsStreaming(false);
    // Final meta line — capture the elapsed time once, use it for both
    // the streaming message meta and (synthesizer) trace entry.
    const finalElapsed = synthStartMsRef.current
      ? ((Date.now() - synthStartMsRef.current) / 1000).toFixed(1) + "s"
      : "0.0s";
    const shortAgent = streamAgentMeta(activeStreamAgentRef.current);
    const streamFinalMeta =
      sourcesCount > 0
        ? `${shortAgent} · ${finalElapsed} · ${sourcesCount} sources`
        : `${shortAgent} · ${finalElapsed} · done`;
    const doneNode = activeStreamAgentRef.current;

    if (sentinel === "[INTERRUPT]") {
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx !== -1) {
          next[idx] = { role: "review", text: draft, id: streamingId, timestamp: next[idx].timestamp };
        }
        return next;
      });
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: "human_review", label: "awaiting…", active: true, time: now() },
      ]);
      setEditingReview(false);
    } else if (sentinel === "[ERROR]") {
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx !== -1) {
          next[idx] = {
            ...next[idx],
            streaming: false,
            text: next[idx].text || "An error occurred. Please try again.",
            error: true,
          };
        }
        return next;
      });
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: doneNode, label: "error", time: now() },
      ]);
    } else {
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx === -1) return next;
        next[idx] = {
          ...next[idx],
          streaming: false,
          meta: streamFinalMeta,
          agent: doneNode,
          // Always write the final draft — rAF may not have flushed the last
          // batch before setIsStreaming fired, so this guarantees the text
          // is never blank in the finished message.
          text: draft || next[idx].text,
        };
        return next;
      });
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: doneNode, label: "done", time: now(), latency: finalElapsed, startMs: null },
      ]);
    }
  }

  // Wrapper used by the input bar's Enter key and Send button. Reads
  // `input` from state — only the welcome chip path passes an explicit
  // text directly to sendMessage.
  function handleSend() {
    sendMessage(input.trim());
    setInput("");
  }

  async function fetchState() {
    const r = await apiFetch(`/threads/${threadId}/state`);
    if (!r.ok) throw new Error(`state fetch failed: ${r.status}`);
    return r.json();
  }

  // Guard against double-clicking Approve / Submit before the previous
  // resume call returns. Without this, a second click re-enters the
  // /review endpoint with no pending interrupt and the backend's
  // `graph.ainvoke(Command(resume=...))` raises a confusing "no
  // checkpoint to resume" error that surfaces in the UI as a
  // fetch failure.
  const [reviewInFlight, setReviewInFlight] = useState(false);

  async function handleApprove() {
    if (reviewInFlight) return;
    setReviewInFlight(true);
    try {
      const res = await apiFetch(`/review/${threadId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "approve" }),
        signal: AbortSignal.timeout(60000),
      });
      if (!res.ok) {
        setStatusError(`review failed (${res.status})`);
        setTimeout(() => setStatusError(null), 3000);
        return;
      }
      const data = await fetchState();
      const finalText = data?.values?.final_response ?? "";
      const agent = agentLabelFromRoute(data?.values?.route);
      setMessages((m) => {
        const next = [...m];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "review") {
            next[i] = {
              role: "agent",
              agent,
              meta: "approved · " + now(),
              text: finalText,
              id: next[i].id,
              timestamp: next[i].timestamp,
            };
            break;
          }
        }
        return next;
      });
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: "human_review", label: "approved", time: now() },
      ]);
      setEditingReview(false);
      setEditText("");
    } finally {
      setReviewInFlight(false);
    }
  }

  function handleEditResend(text) {
    setEditingReview(true);
    setEditText(text);
  }

  async function handleSubmitEdit() {
    if (reviewInFlight) return;
    setReviewInFlight(true);
    try {
      const res = await apiFetch(`/review/${threadId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "edit", edited_response: editText }),
        signal: AbortSignal.timeout(60000),
      });
      if (!res.ok) {
        setStatusError(`review failed (${res.status})`);
        setTimeout(() => setStatusError(null), 3000);
        return;
      }
      const data = await fetchState();
      const finalText = data?.values?.final_response ?? editText;
      const agent = agentLabelFromRoute(data?.values?.route);
      setMessages((m) => {
        const next = [...m];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "review") {
            next[i] = {
              role: "agent",
              agent,
              meta: "edited · " + now(),
              text: finalText,
              id: next[i].id,   // preserve stable key — avoids DOM node swap
              timestamp: next[i].timestamp
            };
            break;
          }
        }
        return next;
      });
      setTrace((t) => [
        ...t.map((e) => (e.active ? { ...e, active: false } : e)),
        { node: "human_review", label: "edited", time: now() },
      ]);
      setEditingReview(false);
    } finally {
      setReviewInFlight(false);
    }
  }

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setStatusError(`File too large (max ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB)`);
      return;
    }
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setStatusError("Only PDF files are supported");
      return;
    }
    const fd = new FormData();
    fd.append("thread_id", threadId);
    fd.append("file", file);
    setIsUploading(true);
    try {
      // `res.ok` covers the 400 (bad thread_id / wrong file type) and 500
      // (ingest failure) cases — without the check, a rejected upload
      // would still log "PDF indexed" because the response body is never
      // read.
      const res = await apiFetch("/upload", { method: "POST", body: fd });
      if (res.ok) {
        setTrace((t) => [...t, { node: "router", label: "PDF indexed", time: now() }]);
      } else {
        setStatusError(`upload failed (${res.status})`);
        setTimeout(() => setStatusError(null), 3000);
      }
    } catch (err) {
      setStatusError(err.message || "upload failed");
      setTimeout(() => setStatusError(null), 3000);
    } finally {
      setIsUploading(false);
    }
    e.target.value = "";
  }

  const [threadList, setThreadList] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Export conversation as Markdown
  function exportConversation() {
    const lines = messages.map((m) => {
      if (m.role === "user") return `**You:** ${m.text}`;
      if (m.role === "agent") return `**${m.agent || "Agent"}:** ${m.text || ""}`;
      if (m.role === "review") return `**Review:** ${m.text || ""}`;
      return "";
    }).filter(Boolean);
    const md = `# AgentFlow Conversation\n_Thread: ${threadId}_\n\n${lines.join("\n\n")}`;
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `agentflow-${threadId.slice(0, 8)}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }

  const fetchThreadList = useCallback(async () => {
    try {
      const res = await apiFetch("/threads");
      if (res.ok) {
        const data = await res.json();
        setThreadList(data.threads || []);
      }
    } catch (err) {
      console.error("Failed to fetch threads", err);
    }
  }, []);

  useEffect(() => {
    if (!showHistory) return;
    fetchThreadList();
    const id = setInterval(fetchThreadList, 30_000);
    return () => clearInterval(id);
  }, [fetchThreadList, showHistory]);

  async function loadThread(id) {
    if (isStreaming) return;
    abortRef.current?.abort();
    abortRef.current = null;
    streamGenRef.current += 1; // invalidate any in-flight stream callbacks

    setThreadId(id);
    setMessages([]);  // clear immediately so stale content never shows
    setTrace([]);
    setIsStreaming(false);
    setShowScrollChip(false);
    setReviewRequired(false);
    setEditingReview(false);
    setEditText("");
    setStatusError(null);
    setShowHistory(false);

    // Generation counter — if the user clicks another thread before
    // either await resolves, discard the stale response.
    const myLoadGen = ++loadGenRef.current;

    try {
      const res = await apiFetch(`/threads/${id}/history`);
      if (myLoadGen !== loadGenRef.current) return;
      if (res.ok) {
        const data = await res.json();
        if (myLoadGen !== loadGenRef.current) return;
        // Give loaded messages stable UI IDs if they lack them
        const loaded = (data.messages || []).map(m => ({
          ...m,
          id: m.id || uuid()
        }));
        setMessages(loaded);
        const hasReviewRow =
          loaded.length > 0 && loaded[loaded.length - 1].role === "review";
        try {
          const stateRes = await apiFetch(`/threads/${id}/state`);
          if (myLoadGen !== loadGenRef.current) return;
          if (stateRes.ok) {
            const stateData = await stateRes.json();
            if (myLoadGen !== loadGenRef.current) return;
            if (stateData?.values?.review_required !== undefined) {
              setReviewRequired(stateData.values.review_required);
            }
            if (stateData?.interrupt?.pending) {
              setReviewRequired(true);
              if (!hasReviewRow) {
                const draft =
                  stateData.interrupt.draft ||
                  stateData?.values?.final_response ||
                  "";
                setMessages((prev) => [
                  ...prev,
                  { role: "review", text: draft, id: uuid() },
                ]);
              }
              setTrace((t) => [
                ...t.map((e) => (e.active ? { ...e, active: false } : e)),
                {
                  node: "human_review",
                  label: "awaiting…",
                  active: true,
                  time: now(),
                },
              ]);
            }
          }
        } catch (e) {
          console.error("Failed to fetch state", e);
          setStatusError("Failed to load thread state");
          setTimeout(() => setStatusError(null), 3000);
        }
      } else {
        // History fetch failed — clear messages so the UI does not keep
        // showing content from a previously loaded thread.
        setMessages([]);
        setStatusError(`Failed to load thread history (${res.status})`);
        setTimeout(() => setStatusError(null), 3000);
      }
    } catch (err) {
      console.error("Failed to load thread history", err);
      setMessages([]);
      setStatusError(err.message || "Failed to load thread history");
      setTimeout(() => setStatusError(null), 3000);
    }
  }

  const handleNewThreadSafe = () => {
    setMessages((currentMessages) => {
      if (currentMessages.length > 0 && !window.confirm("Start a new thread? Your current progress is saved in history.")) {
        return currentMessages;
      }
      resetThread();
      fetchThreadList();
      return currentMessages;
    });
  };

  const inputStyle = {
    ...INPUT_STYLE_BASE,
    resize: "none",
    overflow: "hidden",
    lineHeight: 1.4,
    maxHeight: 140,
    minHeight: 32,
    padding: "6px 12px",
    cursor: isStreaming ? "not-allowed" : "auto",
    opacity: 1,  // readOnly doesn't visually dim; aria-busy carries the state
  };

  // Auto-grow: keep textarea sized to its content up to 6 rows. ResizeObserver
  // would be cleaner but adds a ref dance; this approach uses scrollHeight
  // which is supported everywhere. The `minHeight: 32` on inputStyle
  // prevents zero-height flicker on mount.
  function autoGrow(el) {
    if (!el) return;
    if (autoGrowRafRef.current) cancelAnimationFrame(autoGrowRafRef.current);
    autoGrowRafRef.current = requestAnimationFrame(() => {
      autoGrowRafRef.current = null;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 140) + "px";
    });
  }

  const sendDisabled = isStreaming || !input.trim();

  return (
    <>
      <a href="#chat-input" className="af-skip-link">Skip to chat input</a>
      {statusError && (
        <div style={{
          position: "fixed",
          top: 24,
          right: 24,
          background: "var(--af-bg-panel)",
          border: "1px solid var(--af-status-error)",
          color: "var(--af-status-error)",
          padding: "12px 16px",
          borderRadius: 8,
          boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
          zIndex: 1000,
          fontFamily: "var(--af-font-sans)",
          fontSize: 13,
          fontWeight: 500,
          display: "flex",
          alignItems: "center",
          gap: 8,
          animation: "af-slide-in 0.2s ease-out"
        }}>
          ⚠️ {statusError}
        </div>
      )}
      <div
        style={{
          background: "var(--af-bg-app)",
          height: "100dvh",
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          fontFamily: "var(--af-font-sans)",
        }}
      >
      <div
        style={{
          height: 42,
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 16px",
          borderBottom: "1px solid var(--af-border)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button
            onClick={() => setShowHistory(h => !h)}
            style={{
              background: "transparent",
              border: "none",
              color: showHistory ? "var(--af-text-primary)" : "var(--af-text-muted)",
              fontFamily: "var(--af-font-mono)",
              fontSize: 14,
              cursor: "pointer",
              padding: "4px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}
            title="Toggle History"
            aria-label="Toggle history panel"
          >
            ☰
          </button>
          <span
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 12,
              fontWeight: 500,
              color: "var(--af-text-primary)",
              letterSpacing: "0.04em",
            }}
          >
            AGENTFLOW
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-text-muted)" }} title="Cmd/Ctrl+K for New Thread">
            thread {threadId.slice(0, 6)} <span style={{ opacity: 0.5 }}>⌘K</span>
          </span>
          <button
            onClick={handleNewThreadSafe}
            style={{
              background: "transparent",
              border: "1px solid var(--af-border)",
              borderRadius: 4,
              color: "var(--af-text-muted)",
              padding: "2px 8px",
              fontFamily: "var(--af-font-mono)",
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            New
          </button>
          <button
            onClick={exportConversation}
            disabled={messages.length === 0}
            style={{
              background: "transparent",
              border: "1px solid var(--af-border)",
              borderRadius: 4,
              color: messages.length === 0 ? "var(--af-border)" : "var(--af-text-muted)",
              padding: "2px 8px",
              fontFamily: "var(--af-font-mono)",
              fontSize: 10,
              cursor: messages.length === 0 ? "default" : "pointer",
            }}
            title="Export Conversation"
          >
            Export
          </button>
          <button
            onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
            style={{
              background: "transparent",
              border: "1px solid var(--af-border)",
              borderRadius: 4,
              color: "var(--af-text-muted)",
              padding: "2px 8px",
              fontFamily: "var(--af-font-mono)",
              fontSize: 10,
              cursor: "pointer",
            }}
            title="Toggle theme"
          >
            {theme === "dark" ? "🌞" : "🌙"}
          </button>
          <div style={{ position: "relative" }}>
            <button
              ref={shortcutsButtonRef}
              onClick={() => setShowShortcuts((s) => !s)}
              aria-label="Keyboard shortcuts"
              aria-expanded={showShortcuts}
              aria-haspopup="dialog"
              style={{
                background: "transparent",
                border: "1px solid var(--af-border)",
                borderRadius: "50%",
                color: "var(--af-text-muted)",
                width: 20,
                height: 20,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontFamily: "var(--af-font-mono)",
                fontSize: 10,
                cursor: "pointer",
              }}
            >
              ?
            </button>
            {showShortcuts && (
              <div
                ref={shortcutsPopoverRef}
                role="dialog"
                aria-modal="true"
                aria-label="Keyboard shortcuts"
                style={{
                  position: "absolute",
                  top: 28,
                  right: 0,
                  background: "var(--af-bg-panel)",
                  border: "1px solid var(--af-border)",
                  borderRadius: 6,
                  padding: "12px 16px",
                  zIndex: 10,
                  width: 200,
                  boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                }}
              >
                <div style={{ fontSize: 11, fontFamily: "var(--af-font-mono)", color: "var(--af-text-muted)", marginBottom: 8 }}>SHORTCUTS</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12, color: "var(--af-text-body)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span>Cmd/Ctrl+Enter</span><span style={{ color: "var(--af-text-muted)" }}>Send message</span></div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span>Cmd/Ctrl+K</span><span style={{ color: "var(--af-text-muted)" }}>New thread</span></div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span>Cmd/Ctrl+H</span><span style={{ color: "var(--af-text-muted)" }}>Toggle history</span></div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}><span>Esc</span><span style={{ color: "var(--af-text-muted)" }}>Stop stream / close</span></div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="af-layout" style={{ flex: 1, display: "flex", minHeight: 0, position: "relative" }}>
        
        {showHistory && (
          <div className="af-history-panel">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 14px", borderBottom: "1px solid var(--af-border)", fontSize: 11, fontFamily: "var(--af-font-mono)", color: "var(--af-text-muted)" }}>
              <span>CONVERSATIONS</span>
              <button onClick={() => setShowHistory(false)} style={{ background: "transparent", border: "none", color: "var(--af-text-muted)", cursor: "pointer", fontSize: 12 }} title="Close History">✖</button>
            </div>
            <div style={{ padding: "8px 14px", borderBottom: "1px solid var(--af-border)" }}>
              <input
                type="text"
                placeholder="Search by ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  background: "var(--af-bg-surface)",
                  border: "1px solid var(--af-border)",
                  borderRadius: 4,
                  padding: "6px 8px",
                  color: "var(--af-text-primary)",
                  fontFamily: "var(--af-font-mono)",
                  fontSize: 11,
                  outline: "none",
                }}
              />
            </div>
            <div className="af-scroll" style={{ flex: 1, overflowY: "auto" }}>
              {threadList.filter(t => t.thread_id.toLowerCase().includes(searchQuery.toLowerCase())).length === 0 ? (
                <div style={{ padding: "14px", color: "var(--af-text-muted)", fontSize: 12 }}>No matching conversations.</div>
              ) : (
                threadList
                  .filter(t => t.thread_id.toLowerCase().includes(searchQuery.toLowerCase()))
                  .map((t) => (
                  <div
                    key={t.thread_id}
                    className={`af-history-item ${t.thread_id === threadId ? "active" : ""}`}
                    onClick={() => loadThread(t.thread_id)}
                  >
                    <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 11, color: t.thread_id === threadId ? "var(--af-text-primary)" : "var(--af-text-body)" }}>
                      {t.thread_id.slice(0, 8)}...
                    </div>
                    <div style={{ fontSize: 10, color: "var(--af-text-muted)" }}>
                      {formatLastSeen(t.last_seen)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        <TraceRail trace={trace} />

        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", position: "relative" }}>
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="af-scroll"
            role="log"
            aria-label="Conversation"
            style={{ flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 14 }}
          >
            {messagesWithCitations.length === 0 ? (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "0 20px"
                }}
              >
                <div style={{ textAlign: "center", maxWidth: 500, width: "100%" }}>
                  <div
                    style={{
                      fontFamily: "var(--af-font-mono)",
                      fontSize: 12,
                      color: "var(--af-text-muted)",
                      letterSpacing: "0.08em",
                      marginBottom: 18,
                    }}
                  >
                    AGENTFLOW
                  </div>
                  <div
                    style={{
                      fontFamily: "var(--af-font-sans)",
                      fontSize: 14,
                      color: "var(--af-text-body)",
                      marginBottom: 24,
                    }}
                  >
                    Ask anything. The router decides which agent answers.
                  </div>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 8,
                      alignItems: "center",
                    }}
                  >
                    {[
                      "What are the latest AI research papers?",
                      "Summarize the uploaded PDF",
                      "Write a python script to parse CSV",
                    ].map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        className="af-welcome-chip"
                        onClick={() => sendMessage(prompt)}
                        style={{ width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center" }}
                      >
                        <span>{prompt}</span>
                        <span style={{ opacity: 0.5 }}>→</span>
                      </button>
                    ))}
                  </div>
                  <div style={{ marginTop: 32, fontSize: 13, color: "var(--af-text-muted)", textAlign: "left", background: "var(--af-bg-surface)", padding: 16, borderRadius: 8, lineHeight: 1.5 }}>
                    <strong>💡 Tips:</strong>
                    <ul style={{ paddingLeft: 20, marginTop: 8, marginBottom: 0 }}>
                      <li>Use the <strong>clip icon</strong> to upload a PDF. Ask questions about it and the Analysis agent will answer.</li>
                      <li>AgentFlow has a <strong>Review Mode</strong> for risky actions (like tools that modify data), requiring your approval before proceeding.</li>
                      <li>Hit <strong>Esc</strong> to stop generating at any time.</li>
                    </ul>
                  </div>
                </div>
              </div>
            ) : null}
            {messagesWithCitations.map((msg, i) => (
              <Message
                key={msg.id ?? i}
                msg={msg}
                onApprove={handleApprove}
                onEditResend={handleEditResend}
                onSubmitEdit={handleSubmitEdit}
                editingReview={editingReview}
                editText={editText}
                setEditText={setEditText}
                onRetry={handleRetry}
                onInlineEdit={(text) => {
                  setInput(text);
                  inputRef.current?.focus();
                  autoGrow(inputRef.current);
                }}
              />
            ))}
            {isStreaming && messagesWithCitations.length > 0 && messagesWithCitations[messagesWithCitations.length - 1].role === "user" && (
              <div style={{ maxWidth: "84%", borderLeft: `2px solid var(--af-router)`, padding: "6px 0 6px 12px", opacity: 0.7, marginTop: 16 }}>
                <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-router)", marginBottom: 5 }}>
                  ROUTER · routing
                </div>
                <div className="af-pulse" style={{ color: "var(--af-text-body)", fontSize: 13.5, fontStyle: "italic" }}>
                  Thinking...
                </div>
              </div>
            )}
          </div>

          {/* Hidden live region for screen readers */}
          <div aria-live="polite" style={{ position: "absolute", width: 1, height: 1, padding: 0, margin: -1, overflow: "hidden", clip: "rect(0, 0, 0, 0)", whiteSpace: "nowrap", border: 0 }}>
            {isStreaming ? "Assistant is responding..." : ""}
          </div>

          {showScrollChip && (
            <button
              onClick={() => {
                scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
                setShowScrollChip(false);
                setUnreadCount(0);
              }}
              style={{
                position: "absolute",
                bottom: 60,
                right: 20,
                background: "var(--af-bg-panel)",
                border: "1px solid var(--af-border)",
                borderRadius: 16,
                color: "var(--af-text-primary)",
                padding: "6px 12px",
                fontFamily: "var(--af-font-sans)",
                fontSize: 12,
                cursor: "pointer",
                boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                zIndex: 10,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              ↓ Jump to latest
              {unreadCount > 0 && (
                <span style={{ background: "var(--af-focus)", color: "#fff", padding: "1px 6px", borderRadius: 10, fontSize: 10, fontWeight: 500 }}>
                  {unreadCount}
                </span>
              )}
            </button>
          )}

          {reviewRequired && (
            <div style={{
              background: "var(--af-bg-panel)",
              borderTop: "1px solid var(--af-border)",
              padding: "8px 14px",
              display: "flex",
              alignItems: "center",
              gap: 12,
              flexShrink: 0,
            }}>
              <span style={{ fontSize: 16 }}>🛡️</span>
              <div>
                <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 11, color: "var(--af-human-review, #E74C3C)", fontWeight: "bold" }}>
                  REVIEW MODE ACTIVE
                </div>
                <div style={{ fontSize: 12, color: "var(--af-text-muted)", marginTop: 2 }}>
                  Agent actions will pause and require your explicit approval.
                </div>
              </div>
              <button
                onClick={() => setReviewRequired(false)}
                style={{ marginLeft: "auto", background: "transparent", border: "1px solid var(--af-border)", color: "var(--af-text-primary)", borderRadius: 4, padding: "4px 8px", cursor: "pointer", fontSize: 11, fontFamily: "var(--af-font-sans)" }}
              >
                Disable
              </button>
            </div>
          )}

          <div
            style={{
              flexShrink: 0,
              borderTop: "1px solid var(--af-border)",
              padding: "10px 14px",
              display: "flex",
              gap: 8,
              alignItems: "center",
            }}
          >
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming || isUploading}
              aria-label="Attach PDF"
              style={{
                background: "transparent",
                border: "none",
                color: "var(--af-text-muted)",
                cursor: (isStreaming || isUploading) ? "not-allowed" : "pointer",
                fontSize: 18,
                lineHeight: 1,
                padding: 6,
                opacity: (isStreaming || isUploading) ? 0.5 : 1,
              }}
            >
              {isUploading ? "⏳" : "📎"}
            </button>
            <button
              onClick={() => setReviewRequired((r) => !r)}
              disabled={isStreaming}
              aria-label="Toggle review mode"
              title={reviewRequired ? "Review mode ON — response will pause for approval" : "Review mode OFF"}
              style={{
                background: reviewRequired ? "rgba(227,179,86,0.15)" : "transparent",
                border: reviewRequired ? "1px solid var(--af-review-border)" : "1px solid transparent",
                borderRadius: 5,
                color: reviewRequired ? "var(--af-review)" : "var(--af-text-muted)",
                cursor: isStreaming ? "not-allowed" : "pointer",
                fontFamily: "var(--af-font-mono)",
                fontSize: 10,
                letterSpacing: "0.04em",
                padding: "4px 7px",
                opacity: isStreaming ? 0.5 : 1,
                transition: "all 0.15s",
              }}
            >
              REVIEW
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              aria-label="Upload PDF document"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
            <textarea
              id="chat-input"
              ref={inputRef}
              value={input}
              aria-label="Chat message"
              rows={1}
              readOnly={isStreaming}
              aria-busy={isStreaming}
              onChange={(e) => {
                setInput(e.target.value);
                autoGrow(e.currentTarget);
              }}
              onKeyDown={(e) => {
                // Enter sends; Shift+Enter inserts a newline. IME
                // composition (Chinese / Japanese / Korean) must not be
                // interrupted — `isComposing` and the legacy keyCode
                // 229 both signal "still composing". Without these
                // guards, typing "ni-hao" with pinyin submits the
                // partial composition to the backend on every Enter.
                if (e.key !== "Enter") return;
                if (e.nativeEvent.isComposing || e.keyCode === 229) return;
                if (e.shiftKey) return;
                e.preventDefault();
                handleSend();
              }}
              placeholder="Ask agentflow..."
              style={inputStyle}
            />
            {input.trim().length > 0 && (
              <span className="af-word-count">
                {input.trim().split(/\s+/).length} words
              </span>
            )}
            <button
              onClick={handleSend}
              disabled={sendDisabled}
              aria-label="Send message"
              style={{
                ...SEND_BUTTON_BASE_STYLE,
                cursor: sendDisabled ? "not-allowed" : "pointer",
                opacity: sendDisabled ? 0.5 : 1,
              }}
            >
              ↑
            </button>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
