import { useState, useRef, useEffect, useMemo, useCallback, memo } from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
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

async function apiFetch(path, options = {}) {
  // Inject the JWT as a Bearer header on every call. The backend
  // (backend/auth.py:require_user) resolves identity from this header
  // and scopes thread_ids per-user (backend/main.py:_config_for). If
  // the token is missing or expired the backend returns 401 and the
  // caller is responsible for redirecting to login.
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(apiUrl(path), { ...options, headers });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("agentflow:auth_error"));
  }
  return res;
}

import { MAX_MESSAGE_CHARS, MAX_UPLOAD_BYTES, TRACE_STREAM_NODES, SSE_TOKEN_NODES } from "./constants";
import { parseSSEPayload } from "./sseParser";
import { getToken, clearToken, isExpired, getUsername } from "./auth";
import LoginScreen from "./LoginScreen.jsx";
import useSSE from "./hooks/useSSE";
import MessageBubble from "./components/Chat/MessageBubble";
import ChatInput from "./components/Chat/ChatInput";
import Sidebar from "./components/Sidebar/Sidebar";
import AdminPage from "./pages/AdminPage";

function streamAgentMeta(agent) {
  if (agent === "chat_agent") return "chat";
  if (agent === "research_agent") return "research";
  if (agent === "analysis_agent") return "analysis";
  if (agent === "blog_writer") return "blog";
  if (agent === "memory_reader") return "ltm read";
  if (agent === "memory_writer") return "ltm write";
  if (agent === "stm_compressor") return "stm compress";
  return agent;
}

function agentLabelFromRoute(route) {
  if (route === "chat") return "chat_agent";
  if (route === "research") return "research_agent";
  if (route === "analysis") return "analysis_agent";
  if (route === "blog") return "blog_writer";
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
  blog_writer: "var(--af-blog)",
  memory_reader: "var(--af-memory)",
  memory_writer: "var(--af-memory)",
  stm_compressor: "var(--af-memory)",
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
export default function App() {
  // Auth gate. The /auth/login endpoint is on the backend's
  // PUBLIC_PATHS allowlist, so it works without a token — everything
  // else requires `Authorization: Bearer <jwt>`. We hydrate auth
  // state from localStorage on mount and again every 60s so a token
  // expiring under the user is reflected without a hard reload.
  const [authed, setAuthed] = useState(() => {
    const t = getToken();
    return t && !isExpired(t);
  });
  const [currentUser, setCurrentUser] = useState(() => {
    const t = getToken();
    return t && !isExpired(t) ? getUsername(t) : null;
  });

  useEffect(() => {
    if (!authed) return;
    
    const handleAuthError = () => {
      clearToken();
      setAuthed(false);
      setCurrentUser(null);
    };
    window.addEventListener("agentflow:auth_error", handleAuthError);
    
    const id = setInterval(() => {
      const t = getToken();
      if (!t || isExpired(t)) {
        // Token expired mid-session — drop the user back to login.
        // We don't auto-clear here: a sibling tab may have a fresher
        // token, so we just re-evaluate on the next tick. If still
        // expired on next poll, we bounce.
        const t2 = getToken();
        if (!t2 || isExpired(t2)) {
          handleAuthError();
        }
      }
    }, 60_000);

    // Silent JWT renewal every 20 min (1,200,000 ms)
    const refreshId = setInterval(() => {
      import("./api/client").then(m => m.silentRefresh());
    }, 1_200_000);

    return () => {
      clearInterval(id);
      clearInterval(refreshId);
      window.removeEventListener("agentflow:auth_error", handleAuthError);
    };
  }, [authed]);

  function handleLoginSuccess(token, username) {
    setAuthed(true);
    setCurrentUser(username);
  }

  function handleLogout() {
    clearToken();
    setAuthed(false);
    setCurrentUser(null);
  }

  if (!authed) {
    return <LoginScreen onSuccess={handleLoginSuccess} />;
  }

  return <ChatApp currentUser={currentUser} onLogout={handleLogout} />;
}

function ChatApp({ currentUser, onLogout }) {
  const [threadId, setThreadId] = useState(() => uuid());


    const [theme, setTheme] = useState("dark");
  // Phase 9: blog output and active main tab
  const [blogOutput, setBlogOutput] = useState(null);
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "blog"
  // Phase 9: persistent sidebar (replaces old floating history panel)
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);
    const [input, setInput] = useState("");
    const [isUploading, setIsUploading] = useState(false);
  const [reviewRequired, setReviewRequired] = useState(false);
  const [editingReview, setEditingReview] = useState(false);
  const [editText, setEditText] = useState("");
  const [statusError, setStatusError] = useState(null);
  const statusErrorTimeoutRef = useRef(null);
  const showError = useCallback((msg) => {
    setStatusError(msg);
    if (statusErrorTimeoutRef.current) clearTimeout(statusErrorTimeoutRef.current);
    statusErrorTimeoutRef.current = setTimeout(() => setStatusError(null), 3000);
  }, []);
  const {
    messages,
    setMessages,
    trace,
    setTrace,
    isStreaming,
    setIsStreaming,
    sendMessage,
    resetStreamState,
    abortRef
  } = useSSE({
    threadId,
    showError,
    reviewRequired,
    setReviewRequired,
    setEditingReview,
    activeTab
  });

  const [showScrollChip, setShowScrollChip] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const scrollRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);
        const shortcutsButtonRef = useRef(null);
  const shortcutsPopoverRef = useRef(null);
  const autoGrowRafRef = useRef(null);
  // Per-stream generation counter. Incremented at the start of every
  // sendMessage call; every async callback captures its own snapshot and
  // bails out if the ref has moved on (i.e. a newer request has started
  // or resetThread was called). This prevents a stale fetch/watchdog/rAF
  // from mutating state that now belongs to a different conversation.
    const loadGenRef = useRef(0);
    // Phase 2: stream timing. `synthStartMs` is captured the moment the
  // streaming message appears in the UI; `lastMetaUpdateMs` gates the
  // rAF's meta write to 4 Hz so a 60Hz rAF doesn't re-render the
  // message list on every frame just to update a single label.
      // Per-stream source count. Backend emits [SOURCES:n] right before
  // [DONE] only on a non-interrupt run; the meta line picks it up.
    // Watchdog: timestamp of the last received token. Reset to Date.now()
  // on every successful read. Interval checks every 5s and triggers an
  // error state if the gap exceeds STALL_MS (60s).
  
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
      if (msg.role !== "agent" || msg.error || msg.streaming) return msg;
      // Parse citations for synthesizer (final polished answer with Sources block)
      // and research_agent (raw output may include inline [1] url references).
      if (msg.agent !== "synthesizer" && msg.agent !== "research_agent") return msg;
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
    resetStreamState();
    setThreadId(uuid());
    setInput("");
    setShowScrollChip(false);
    setReviewRequired(false);
    setEditingReview(false);
    setEditText("");
    setStatusError(null);
    setBlogOutput(null);
    setActiveTab("chat");
  }, [resetStreamState]);

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
        setSidebarOpen((v) => !v);
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
  

  // Wrapper used by the input bar's Enter key and Send button. Reads
  // `input` from state — only the welcome chip path passes an explicit
  // text directly to sendMessage.
  function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    sendMessage(trimmed);
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
        showError(`review failed (${res.status})`);
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
        showError(`review failed (${res.status})`);
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
        let stats = null;
        try { stats = await res.json(); } catch (_) {}
        const chunks = stats?.chunks ?? "?";
        const pages  = stats?.pages  ?? "?";
        const label  = `📄 **${file.name}** indexed — ${pages} page(s), ${chunks} chunk(s). You can now ask questions about it.`;
        setMessages((m) => [
          ...m,
          { role: "agent", agent: "router", text: label, id: crypto.randomUUID() },
        ]);
        setTrace((t) => [...t, { node: "router", label: "PDF indexed", time: now() }]);
      } else {
        let detail = "";
        try { const body = await res.json(); detail = body?.detail || ""; } catch (_) {}
        showError(`Upload failed (${res.status})${detail ? ": " + detail : ""}`);
      }
    } catch (err) {
      showError(err.message || "upload failed");
    } finally {
      setIsUploading(false);
    }
    e.target.value = "";
  }

  const [threadList, setThreadList] = useState([]);
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

  // Auto-refresh thread list whenever sidebar is open.
  useEffect(() => {
    if (!sidebarOpen) return;
    fetchThreadList();
    const id = setInterval(fetchThreadList, 30_000);
    return () => clearInterval(id);
  }, [fetchThreadList, sidebarOpen]);

  const wasStreamingRef = useRef(false);
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming) {
      fetchThreadList();
      // If the completed stream was a blog generation, fetch and display the
      // blog output from backend state (blog_output is stored in LangGraph state,
      // not streamed directly to the frontend).
      const lastMsg = messages[messages.length - 1];
      const isBlogStream = lastMsg?.role === "agent" && lastMsg?.agent === "blog_writer";
      if (isBlogStream) {
        apiFetch(`/threads/${threadId}/blog`)
          .then(r => r.ok ? r.json() : null)
          .then(data => {
            if (data?.blog_output) {
              setBlogOutput(data.blog_output);
              setActiveTab("blog");
            }
          })
          .catch(() => {});
      }
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming, fetchThreadList, messages, threadId]);

  async function deleteThread(id, e) {
    e.stopPropagation();
    if (!window.confirm(`Delete thread ${id.slice(0, 8)}…? This is irreversible.`)) return;
    try {
      const res = await apiFetch(`/threads/${id}`, { method: "DELETE" });
      if (res.ok) {
        if (id === threadId) {
          resetThread();
        }
        fetchThreadList();
      } else {
        showError("Failed to delete thread");
      }
    } catch (err) {
      console.error("deleteThread failed", err);
      showError("Failed to delete thread");
    }
  }

  async function loadThread(id) {
    if (isStreaming) return;
    resetStreamState();
    setThreadId(id);
    setShowScrollChip(false);
    setReviewRequired(false);
    setEditingReview(false);
    setEditText("");
    setStatusError(null);
    setBlogOutput(null);
    setActiveTab("chat");

    // Generation counter — if the user clicks another thread before
    // either await resolves, discard the stale response.
    const myLoadGen = ++loadGenRef.current;

    try {
      const [histRes, blogRes] = await Promise.all([
        apiFetch(`/threads/${id}/history`),
        apiFetch(`/threads/${id}/blog`),
      ]);
      if (myLoadGen !== loadGenRef.current) return;

      // ── Blog output ──────────────────────────────────────────────────────
      if (blogRes.ok) {
        try {
          const blogData = await blogRes.json();
          if (myLoadGen !== loadGenRef.current) return;
          if (blogData?.blog_output) {
            setBlogOutput(blogData.blog_output);
            setActiveTab("blog");   // switch to Blog tab so user sees it
          }
        } catch { /* non-JSON or empty — ignore */ }
      }

      // ── Chat history ─────────────────────────────────────────────────────
      if (!histRes.ok) {
        setMessages([]);
        showError(`Failed to load thread history (${histRes.status})`);
        return;
      }

      const data = await histRes.json();
      if (myLoadGen !== loadGenRef.current) return;

      const loaded = (data.messages || []).map(m => ({
        ...m,
        id: m.id || uuid(),
      }));
      setMessages(loaded);

      // ── Reconstruct execution trace from history messages ─────────────────
      // We can't restore exact timing, but we can show which agents ran.
      const reconstructedTrace = loaded
        .filter(m => m.role === "agent" && m.agent)
        .map(m => ({
          node: m.agent,
          label: "done",
          active: false,
          time: m.timestamp || null,
          latency: null,
        }));
      if (reconstructedTrace.length > 0) {
        setTrace(reconstructedTrace);
      }

      // ── State / interrupt ─────────────────────────────────────────────────
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
              setMessages(prev => [
                ...prev,
                { role: "review", text: draft, id: uuid() },
              ]);
            }
            setTrace(t => [
              ...t.map(e => (e.active ? { ...e, active: false } : e)),
              { node: "human_review", label: "awaiting…", active: true, time: now() },
            ]);
          }
        }
      } catch (e) {
        console.error("Failed to fetch state", e);
      }
    } catch (err) {
      if (myLoadGen !== loadGenRef.current) return;
      console.error("Failed to load thread", err);
      setMessages([]);
      showError(err.message || "Failed to load thread history");
    }
  }

  const handleNewThreadSafe = () => {
    if (messages.length > 0 && !window.confirm("Start a new thread? Your current progress is saved in history.")) {
      return;
    }
    resetThread();
    fetchThreadList();
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
            onClick={() => setSidebarOpen(h => !h)}
            style={{
              background: "transparent",
              border: "none",
              color: sidebarOpen ? "var(--af-text-primary)" : "var(--af-text-muted)",
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
          {currentUser && (
            <span
              style={{
                fontFamily: "var(--af-font-mono)",
                fontSize: 10.5,
                color: "var(--af-text-muted)",
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
              title={`signed in as ${currentUser}`}
            >
              <span style={{ opacity: 0.5 }}>·</span>
              <span>{currentUser}</span>
              <button
                onClick={onLogout}
                aria-label="Sign out"
                title="Sign out"
                style={{
                  background: "transparent",
                  border: "1px solid var(--af-border)",
                  borderRadius: 4,
                  color: "var(--af-text-muted)",
                  padding: "1px 7px",
                  fontFamily: "var(--af-font-mono)",
                  fontSize: 9.5,
                  cursor: "pointer",
                  letterSpacing: "0.04em",
                }}
              >
                SIGN OUT
              </button>
            </span>
          )}
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
        
        {/* Persistent thread sidebar (replaces old floating history panel) */}
        <Sidebar
          sidebarOpen={sidebarOpen}
          threadList={threadList}
          threadId={threadId}
          loadThread={loadThread}
          deleteThread={deleteThread}
          handleNewThreadSafe={handleNewThreadSafe}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
        />

        <TraceRail trace={trace} />

        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", position: "relative" }}>
          {/* Tab bar: Chat | Blog | Admin */}
          <div className="af-blog-tab-bar">
            <button
              className={`af-tab-btn ${activeTab === "chat" ? "active" : ""}`}
              onClick={() => setActiveTab("chat")}
            >
              💬 Chat
            </button>
            <button
              className={`af-tab-btn ${activeTab === "blog" ? "active" : ""}`}
              onClick={() => setActiveTab("blog")}
              style={{ position: "relative" }}
            >
              ✍️ Blog
              {blogOutput && activeTab !== "blog" && (
                <span style={{
                  position: "absolute", top: 4, right: 4,
                  width: 6, height: 6, borderRadius: "50%",
                  background: "var(--af-blog)",
                }} />
              )}
            </button>
            {currentUser === "admin" && (
              <button
                className={`af-tab-btn ${activeTab === "admin" ? "active" : ""}`}
                onClick={() => setActiveTab("admin")}
              >
                ⚙️ Admin
              </button>
            )}
          </div>

          {/* Blog viewer tab */}
          {activeTab === "blog" && (
            <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
              {blogOutput ? (
                <div className="af-blog-viewer af-scroll af-blog-enter">
                  <div className="af-blog-title">{blogOutput.title || "Blog Post"}</div>
                  {blogOutput.meta_description && (
                    <div style={{ fontSize: 13, color: "var(--af-text-muted)", marginBottom: 14, lineHeight: 1.5, fontStyle: "italic" }}>
                      {blogOutput.meta_description}
                    </div>
                  )}
                  {blogOutput.tags?.length > 0 && (
                    <div className="af-blog-meta">
                      {blogOutput.tags.map(tag => (
                        <span key={tag} className="af-blog-tag">{tag}</span>
                      ))}
                    </div>
                  )}
                  {(blogOutput.sections || []).map((section, i) => (
                    <div key={i}>
                      {section.heading && (
                        <div className="af-blog-section-heading">{section.heading}</div>
                      )}
                      <div className="af-blog-section-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
                          {section.content || ""}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ))}
                  <div className="af-blog-actions">
                    <button
                      className="af-blog-action-btn"
                      onClick={() => {
                        const md = [
                          `# ${blogOutput.title}`,
                          blogOutput.meta_description ? `*${blogOutput.meta_description}*` : "",
                          ...(blogOutput.sections || []).map(s =>
                            `## ${s.heading}\n\n${s.content}`
                          ),
                        ].filter(Boolean).join("\n\n");
                        const blob = new Blob([md], { type: "text/markdown" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `${(blogOutput.title || "blog").toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "")}.md`;
                        a.click();
                        setTimeout(() => URL.revokeObjectURL(url), 100);
                      }}
                    >
                      ⬇️ Download Markdown
                    </button>
                    <button
                      className="af-blog-action-btn"
                      onClick={() => {
                        const text = (blogOutput.sections || []).map(s =>
                          `${s.heading}\n\n${s.content}`
                        ).join("\n\n");
                        navigator.clipboard.writeText(text).catch(() => {});
                      }}
                    >
                      📋 Copy Text
                    </button>
                  </div>
                </div>
              ) : (
                <div className="af-blog-empty">
                  <div className="af-blog-empty-icon">✍️</div>
                  <div>No blog post yet.</div>
                  <div style={{ fontSize: 11, opacity: 0.7 }}>Ask AgentFlow to write a blog post.</div>
                  <button
                    className="af-welcome-chip"
                    onClick={() => { setActiveTab("chat"); sendMessage("Write a blog post about the future of AI and large language models."); }}
                    style={{ marginTop: 12 }}
                  >
                    Try: Write a blog post about AI →
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Admin tab */}
          {activeTab === "admin" && (
            <AdminPage />
          )}

          {/* Chat tab */}
          {activeTab === "chat" && (
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
                      "Write a blog post about the future of AI",
                      "Calculate compound interest on $5000 at 4% for 10 years",
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
              <MessageBubble
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
          )} {/* end activeTab === "chat" */}

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

          <ChatInput
            input={input}
            setInput={setInput}
            isStreaming={isStreaming}
            isUploading={isUploading}
            reviewRequired={reviewRequired}
            setReviewRequired={setReviewRequired}
            handleSend={handleSend}
            handleFileChange={handleFileChange}
            inputRef={inputRef}
          />
        </div>
      </div>
    </div>
    </>
  );
}
