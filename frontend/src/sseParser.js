/**
 * Parse a single SSE `data:` payload line from /chat.
 * Returns { kind, value } where kind is one of:
 *   token | done | interrupt | error | sources | final | node_start | node_end | tool_start | skip
 */

export function parseSSEPayload(rawPayload) {
  const payload = rawPayload.trim();

  if (payload === "[DONE]") return { kind: "done" };
  if (payload === "[INTERRUPT]") return { kind: "interrupt" };
  if (payload.startsWith("[ERROR]")) return { kind: "error", value: payload };

  if (payload.startsWith("[SOURCES:")) {
    const n = parseInt(payload.slice(9, -1), 10);
    if (Number.isFinite(n) && n >= 0) return { kind: "sources", value: n };
    return { kind: "skip" };
  }

  if (payload.startsWith("[FINAL:")) {
    try {
      return { kind: "final", value: JSON.parse(payload.slice(7, -1)) };
    } catch {
      return { kind: "skip" };
    }
  }

  if (payload.startsWith("[TOOL_START:")) {
    return { kind: "tool_start", value: payload.slice(12, -1) };
  }

  if (payload.startsWith("[NODE_START:")) {
    const inner = payload.slice(12, -1);
    const pipeIdx = inner.indexOf("|");
    const node = pipeIdx === -1 ? inner : inner.slice(0, pipeIdx);
    let startMs = Date.now();
    if (pipeIdx !== -1) {
      const tSeg = inner.slice(pipeIdx + 1);
      const eqIdx = tSeg.indexOf("=");
      if (eqIdx !== -1) {
        const parsed = Date.parse(tSeg.slice(eqIdx + 1));
        if (Number.isFinite(parsed)) startMs = parsed;
      }
    }
    return { kind: "node_start", value: { node, startMs } };
  }

  if (payload.startsWith("[NODE_END:")) {
    return { kind: "node_end", value: payload.slice(10, -1) };
  }

  return { kind: "token", value: rawPayload };
}
