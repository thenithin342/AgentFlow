import { useState, useRef, useCallback, useEffect } from "react";
import { apiFetch } from "../api/client";
import { parseSSEPayload } from "../sseParser";
import { SSE_TOKEN_NODES } from "../constants";
import { uuid, now, streamAgentMeta } from "../utils";

export default function useSSE({ threadId, showError, reviewRequired, setEditingReview }) {
  const [messages, setMessages] = useState([]);
  const [trace, setTrace] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stallHint, setStallHint] = useState(null);
  const [routerFallback, setRouterFallback] = useState(false);

  const clearRouterFallback = useCallback(() => setRouterFallback(false), []);

  const abortRef = useRef(null);
  const pendingDraftRef = useRef(null);
  const rafPendingRef = useRef(false);
  
  const streamGenRef = useRef(0);
  const activeStreamAgentRef = useRef("router");
  
  const synthStartMsRef = useRef(null);
  const lastMetaUpdateMsRef = useRef(0);
  const streamSourcesRef = useRef(0);
  const lastTokenAtMsRef = useRef(0);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  const resetStreamState = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    streamGenRef.current += 1;
    setMessages([]);
    setTrace([]);
    setIsStreaming(false);
    setStallHint(null);
    setRouterFallback(false);
  }, []);

  const sendMessage = useCallback(async (text) => {
    if (!text || !text.trim() || isStreaming) return;
    
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();
    
    const myGen = streamGenRef.current + 1;
    streamGenRef.current = myGen;
    
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    setMessages((m) => [...m, { role: "user", text, id: uuid(), timestamp }]);
    setTrace((t) => [...t, { node: "router", label: "routing…", active: true, time: now() }]);
    setIsStreaming(true);
    setStallHint(null);
    setRouterFallback(false);
    
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
        setMessages((m) => {
          if (m.length === 0) return m;
          const next = [...m];
          const last = next[next.length - 1];
          if (last.role === "agent" && last.streaming) {
            next[next.length - 1] = { ...last, streaming: false, aborted: true, meta: "aborted" };
          } else if (last.role === "user") {
            next.push({ role: "agent", agent: "router", meta: "aborted", text: "[aborted before first token]", aborted: true, id: uuid() });
          }
          return next;
        });
        setTrace((t) => [...t.map((x) => (x.active ? { ...x, active: false } : x)), { node: "router", label: "aborted", time: now() }]);
        return;
      }
      setIsStreaming(false);
      setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: "router", label: "error", time: now() }]);
      showError(err.message || "network error");
      return;
    }

    if (!res.ok) {
      setIsStreaming(false);
      setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: "router", label: `error ${res.status}`, time: now() }]);
      let detail = `server returned ${res.status}`;
      try {
        const errBody = await res.json();
        if (errBody?.detail) detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
      } catch { /* non-JSON */ }
      showError(detail);
      return;
    }

    if (!res.body) {
      setIsStreaming(false);
      setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: "router", label: "error (no body)", time: now() }]);
      showError("Server returned a response with no body (streaming not supported).");
      return;
    }

    const streamingId = uuid();
    const agentTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    setMessages((m) => [
      ...m,
      { role: "agent", agent: "router", meta: "routing…", text: "", streaming: true, id: streamingId, timestamp: agentTimestamp },
    ]);
    
    synthStartMsRef.current = Date.now();
    lastMetaUpdateMsRef.current = Date.now();
    lastTokenAtMsRef.current = Date.now();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let draft = "";
    let sentinel = null;
    let sentinelDetail = "";
    let sourcesCount = 0;
    // SSE events: backend splits one token containing "\n" into multiple
    // `data:` lines terminated by a blank line. Those lines belong to ONE
    // event and must be rejoined with "\n" — treating each line as its own
    // token silently drops newlines and collapses markdown tables/headings
    // into a single paragraph.
    let pendingDataLines = [];

    const handlePayload = (rawPayload) => {
      const parsed = parseSSEPayload(rawPayload);

      // Defensive: backend may surface the router LLM failure as a
      // node_update/graph payload containing {"router_fallback": true}
      // (see backend/graph/router.py). Catch it regardless of event kind.
      try {
        if (typeof rawPayload === "string" && rawPayload.includes("router_fallback")) {
          const maybe = JSON.parse(rawPayload.slice(rawPayload.indexOf("{")));
          if (maybe && maybe.router_fallback === true) {
            if (streamGenRef.current === myGen) setRouterFallback(true);
          } else if (/["']?router_fallback["']?\s*[:=]\s*true/i.test(rawPayload)) {
            if (streamGenRef.current === myGen) setRouterFallback(true);
          }
        }
        if (parsed && parsed.value && typeof parsed.value === "object" && parsed.value.router_fallback === true) {
          if (streamGenRef.current === myGen) setRouterFallback(true);
        }
      } catch { /* non-JSON payload — ignore */ }

      if (parsed.kind === "done") { sentinel = "[DONE]"; return true; }
      if (parsed.kind === "interrupt") { sentinel = "[INTERRUPT]"; return true; }
      if (parsed.kind === "error") { sentinel = "[ERROR]"; sentinelDetail = (parsed.value || "").replace(/^\[ERROR\]\s*/, ""); return true; }
      if (parsed.kind === "sources") { sourcesCount = parsed.value; streamSourcesRef.current = parsed.value; return false; }
      if (parsed.kind === "final") { if (!draft) { draft = parsed.value; pendingDraftRef.current = draft; } return false; }
      if (parsed.kind === "fallback") {
        // Router LLM failed — add a subtle trace entry but don’t block the response.
        if (streamGenRef.current === myGen) setRouterFallback(true);
        setTrace((t) => [...t, { node: "router", label: "degraded (chat fallback)", time: now() }]);
        return false;
      }
      if (parsed.kind === "tool_start") {
        setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: parsed.value, label: "tool…", active: true, time: now() }]);
        return false;
      }
      if (parsed.kind === "node_start") {
        const { node, startMs } = parsed.value;
        setTrace((t) => {
          if (t.length > 0 && t[t.length - 1].node === node) return t;
          const next = t.map((e) => (e.active ? { ...e, active: false } : e));
          next.push({ node, label: "working…", active: true, time: now(), startMs });
          return next;
        });
        if (SSE_TOKEN_NODES.has(node)) {
          activeStreamAgentRef.current = node;
          setMessages((m) => {
            const next = [...m];
            const idx = next.findIndex((x) => x.id === streamingId);
            if (idx === -1 || !next[idx].streaming) return next;
            next[idx] = { ...next[idx], agent: node, meta: `${streamAgentMeta(node)} · working…` };
            return next;
          });
        }
        return false;
      }
      if (parsed.kind === "node_end") {
        const node = parsed.value;
        setTrace((t) => {
          for (let i = t.length - 1; i >= 0; i--) {
            if (t[i].node === node && t[i].startMs) {
              const ms = Date.now() - t[i].startMs;
              const next = [...t];
              next[i] = { ...t[i], label: "done", active: false, latency: (ms / 1000).toFixed(1) + "s", startMs: null };
              return next;
            }
          }
          return t;
        });
        return false;
      }
      if (parsed.kind === "skip") return false;

      // Only advance the stall watchdog on real token output — not on SSE
      // keep-alive comments (`: keep-alive`) which arrive as empty chunks
      // and would mask genuine LLM stalls if they reset lastTokenAtMsRef.
      lastTokenAtMsRef.current = Date.now();

      draft += parsed.value;
      pendingDraftRef.current = draft;
      if (!rafPendingRef.current) {
        rafPendingRef.current = true;
        requestAnimationFrame(() => {
          rafPendingRef.current = false;
          if (streamGenRef.current !== myGen) return;
          const d = pendingDraftRef.current;
          const nowMs = Date.now();
          const metaDue = nowMs - lastMetaUpdateMsRef.current >= 250;
          const startMs = synthStartMsRef.current ?? nowMs;
          const elapsed = ((nowMs - startMs) / 1000).toFixed(1) + "s";
          const short = streamAgentMeta(activeStreamAgentRef.current);
          setMessages((m) => {
            const next = [...m];
            const idx = next.findIndex((x) => x.id === streamingId);
            if (idx === -1) return next;
            next[idx] = { ...next[idx], text: d, ...(metaDue ? { meta: `${short} · ${elapsed}` } : {}) };
            return next;
          });
          if (metaDue) lastMetaUpdateMsRef.current = nowMs;
        });
      }
      return false;
    };

    const STALL_MS = 60_000;
    const watchdog = setInterval(() => {
      if (streamGenRef.current !== myGen) { clearInterval(watchdog); return; }
      const elapsed = Date.now() - lastTokenAtMsRef.current;
      if (elapsed > 15_000 && elapsed <= 30_000) {
        if (streamGenRef.current === myGen) setStallHint("Still working…");
      } else if (elapsed > 30_000 && elapsed <= 55_000) {
        if (streamGenRef.current === myGen) setStallHint("Taking longer than usual — complex query…");
      }
      if (elapsed > STALL_MS) {
        sentinel = "[ERROR]";
        clearInterval(watchdog);
        try { reader.cancel(); } catch { /* cancel may throw if already closed — ignore */ }
        if (streamGenRef.current !== myGen) return;
        setMessages((m) => {
          const next = [...m];
          const idx = next.findIndex((x) => x.id === streamingId);
          if (idx !== -1 && next[idx].streaming) {
            next[idx] = { ...next[idx], streaming: false, error: true, text: next[idx].text || "Stream stalled (60s with no tokens)." };
          }
          return next;
        });
        setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: "synthesizer", label: "stalled", time: now() }]);
        setIsStreaming(false);
        setStallHint(null);
      }
    }, 5_000);

    try {
      while (!sentinel) {
        const { done, value } = await reader.read();
        if (done) break;
        lastTokenAtMsRef.current = Date.now();
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const rawLine of lines) {
          const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
          if (line.startsWith("data: ")) {
            pendingDataLines.push(line.slice(6));
            continue;
          }
          if (line === "") {
            // Blank line = end of one SSE event. Rejoin multi-line data.
            if (pendingDataLines.length === 0) continue;
            const rawPayload = pendingDataLines.join("\n");
            pendingDataLines = [];
            if (handlePayload(rawPayload)) break;
            if (sentinel) break;
            continue;
          }
          // Ignore SSE comments (e.g. ": keep-alive") and unknown fields.
        }
        if (sentinel) break;
      }
      // Flush a trailing event not terminated by a blank line.
      if (!sentinel && pendingDataLines.length > 0) {
        handlePayload(pendingDataLines.join("\n"));
        pendingDataLines = [];
      }
    } catch (err) {
      if (err.name === "AbortError") {
        if (streamGenRef.current === myGen) { setIsStreaming(false); setStallHint(null); }
        return;
      }
      if (streamGenRef.current !== myGen) return;
      setIsStreaming(false);
      setStallHint(null);
      showError(err.message || "stream error");
      return;
    } finally {
      clearInterval(watchdog);
    }

    if (streamGenRef.current !== myGen) return;
    setIsStreaming(false);
    setStallHint(null);
    
    const finalElapsed = synthStartMsRef.current ? ((Date.now() - synthStartMsRef.current) / 1000).toFixed(1) + "s" : "0.0s";
    const shortAgent = streamAgentMeta(activeStreamAgentRef.current);
    const streamFinalMeta = sourcesCount > 0 ? `${shortAgent} · ${finalElapsed} · ${sourcesCount} sources` : `${shortAgent} · ${finalElapsed} · done`;
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
      setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: "human_review", label: "awaiting…", active: true, time: now() }]);
      setEditingReview(false);
    } else if (sentinel === "[ERROR]") {
      const detail = (sentinelDetail || "").trim();
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx !== -1) {
          next[idx] = { ...next[idx], streaming: false, text: detail || next[idx].text || "An error occurred. Please try again.", error: true };
        }
        return next;
      });
      setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: doneNode, label: "error", time: now() }]);
    } else {
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx !== -1) {
          const finalText = pendingDraftRef.current || next[idx].text || "";
          next[idx] = { ...next[idx], streaming: false, meta: streamFinalMeta, text: finalText };
        }
        return next;
      });
      setTrace((t) => {
        for (let i = t.length - 1; i >= 0; i--) {
          if (t[i].node === doneNode && t[i].startMs) {
            const ms = Date.now() - t[i].startMs;
            const next = [...t];
            next[i] = { ...t[i], label: "done", active: false, latency: (ms / 1000).toFixed(1) + "s", startMs: null };
            return next;
          }
        }
        return t.map((e) => (e.active ? { ...e, active: false } : e));
      });
    }
  }, [isStreaming, threadId, reviewRequired, showError, setEditingReview]);

  return {
    messages,
    setMessages,
    trace,
    setTrace,
    isStreaming,
    setIsStreaming,
    sendMessage,
    resetStreamState,
    abortRef,
    stallHint,
    routerFallback,
    clearRouterFallback
  };
}
