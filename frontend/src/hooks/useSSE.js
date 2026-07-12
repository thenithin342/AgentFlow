import { useState, useRef, useCallback, useEffect } from "react";
import { apiFetch } from "../api/client";
import { parseSSEPayload } from "../sseParser";
import { SSE_TOKEN_NODES } from "../constants";
import { uuid, now, streamAgentMeta } from "../utils";

export default function useSSE({ threadId, showError, reviewRequired, setReviewRequired, setEditingReview, activeTab }) {
  const [messages, setMessages] = useState([]);
  const [trace, setTrace] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

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
    let sourcesCount = 0;

    const STALL_MS = 60_000;
    const watchdog = setInterval(() => {
      if (streamGenRef.current !== myGen) { clearInterval(watchdog); return; }
      if (Date.now() - lastTokenAtMsRef.current > STALL_MS) {
        sentinel = "[ERROR]";
        clearInterval(watchdog);
        try { reader.cancel(); } catch {}
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

          if (parsed.kind === "done") { sentinel = "[DONE]"; break; }
          if (parsed.kind === "interrupt") { sentinel = "[INTERRUPT]"; break; }
          if (parsed.kind === "error") { sentinel = "[ERROR]"; break; }
          if (parsed.kind === "sources") { sourcesCount = parsed.value; streamSourcesRef.current = parsed.value; continue; }
          if (parsed.kind === "final") { if (!draft) { draft = parsed.value; pendingDraftRef.current = draft; } continue; }
          if (parsed.kind === "tool_start") {
            setTrace((t) => [...t.map((e) => (e.active ? { ...e, active: false } : e)), { node: parsed.value, label: "tool…", active: true, time: now() }]);
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
            continue;
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
            continue;
          }
          if (parsed.kind === "skip") continue;

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
      showError(err.message || "stream error");
      return;
    } finally {
      clearInterval(watchdog);
    }

    if (streamGenRef.current !== myGen) return;
    setIsStreaming(false);
    
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
      setMessages((m) => {
        const next = [...m];
        const idx = next.findIndex((x) => x.id === streamingId);
        if (idx !== -1) {
          next[idx] = { ...next[idx], streaming: false, text: next[idx].text || "An error occurred. Please try again.", error: true };
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
  }, [isStreaming, threadId, reviewRequired, activeTab, showError, setEditingReview]);

  return {
    messages,
    setMessages,
    trace,
    setTrace,
    isStreaming,
    setIsStreaming,
    sendMessage,
    resetStreamState,
    abortRef
  };
}
