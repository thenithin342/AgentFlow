/**
 * sseParser.test.js — unit tests for parseSSEPayload
 *
 * Covers:
 *   - All sentinel kinds: done, interrupt, error, sources, final,
 *     node_start (with + without timestamp), node_end, tool_start
 *   - Token passthrough (including markdown, multiline, whitespace)
 *   - Malformed / partial sentinels fall through to token
 *   - [FINAL:] with invalid JSON falls through to skip
 *   - [SOURCES:] with non-numeric content falls through to skip
 *   - [NODE_START:] with and without the |t= segment
 *   - Empty string and whitespace-only inputs
 *   - Model accidentally emitting a sentinel literal as content
 */

import { describe, expect, test } from "vitest";
import { parseSSEPayload } from "../src/sseParser";

// ---------------------------------------------------------------------------
// Terminal sentinels
// ---------------------------------------------------------------------------
describe("done sentinel", () => {
  test("[DONE] returns kind:done", () => {
    expect(parseSSEPayload("[DONE]")).toEqual({ kind: "done" });
  });

  test("[DONE] with surrounding whitespace returns kind:done (trim)", () => {
    expect(parseSSEPayload("  [DONE]  ")).toEqual({ kind: "done" });
  });
});

describe("interrupt sentinel", () => {
  test("[INTERRUPT] returns kind:interrupt", () => {
    expect(parseSSEPayload("[INTERRUPT]")).toEqual({ kind: "interrupt" });
  });
});

describe("fallback sentinel", () => {
  test("[FALLBACK] returns kind:fallback", () => {
    expect(parseSSEPayload("[FALLBACK]")).toEqual({ kind: "fallback" });
  });

  test("[FALLBACK] with surrounding whitespace returns kind:fallback", () => {
    expect(parseSSEPayload("  [FALLBACK]  ")).toEqual({ kind: "fallback" });
  });

  test("sentence containing [FALLBACK] literally is a token", () => {
    const r = parseSSEPayload("The model returned [FALLBACK] mode.");
    expect(r.kind).toBe("token");
  });
});

describe("error sentinel", () => {
  test("[ERROR] bare returns kind:error", () => {
    const r = parseSSEPayload("[ERROR]");
    expect(r.kind).toBe("error");
    expect(r.value).toBe("[ERROR]");
  });

  test("[ERROR] with detail returns kind:error with full value", () => {
    const payload = "[ERROR] ValueError: something went wrong";
    const r = parseSSEPayload(payload);
    expect(r.kind).toBe("error");
    expect(r.value).toBe(payload);
  });

  test("[ERROR] with colon-less message", () => {
    const r = parseSSEPayload("[ERROR] timeout");
    expect(r.kind).toBe("error");
    expect(r.value).toContain("timeout");
  });
});

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------
describe("sources sentinel", () => {
  test("[SOURCES:0] returns kind:sources, value:0", () => {
    expect(parseSSEPayload("[SOURCES:0]")).toEqual({ kind: "sources", value: 0 });
  });

  test("[SOURCES:3] returns kind:sources, value:3", () => {
    expect(parseSSEPayload("[SOURCES:3]")).toEqual({ kind: "sources", value: 3 });
  });

  test("[SOURCES:abc] returns kind:skip (non-numeric)", () => {
    expect(parseSSEPayload("[SOURCES:abc]")).toEqual({ kind: "skip" });
  });

  test("[SOURCES:-1] returns kind:skip (negative)", () => {
    // -1 is not matched by /^\d+$/ so falls to skip
    expect(parseSSEPayload("[SOURCES:-1]")).toEqual({ kind: "skip" });
  });

  test("[SOURCES:] empty returns kind:skip", () => {
    expect(parseSSEPayload("[SOURCES:]")).toEqual({ kind: "skip" });
  });
});

// ---------------------------------------------------------------------------
// Final
// ---------------------------------------------------------------------------
describe("final sentinel", () => {
  test("[FINAL:\"hello\"] returns kind:final, value:hello", () => {
    expect(parseSSEPayload('[FINAL:"hello"]')).toEqual({ kind: "final", value: "hello" });
  });

  test("[FINAL:{...}] returns kind:final with parsed JSON", () => {
    const r = parseSSEPayload('[FINAL:{"key":"val"}]');
    expect(r.kind).toBe("final");
    expect(r.value).toEqual({ key: "val" });
  });

  test("[FINAL:invalid json] returns kind:skip", () => {
    expect(parseSSEPayload("[FINAL:not json]")).toEqual({ kind: "skip" });
  });

  test("[FINAL:] empty body returns kind:skip", () => {
    expect(parseSSEPayload("[FINAL:]")).toEqual({ kind: "skip" });
  });
});

// ---------------------------------------------------------------------------
// Tool start
// ---------------------------------------------------------------------------
describe("tool_start sentinel", () => {
  test("[TOOL_START:retrieve_documents] returns kind:tool_start", () => {
    expect(parseSSEPayload("[TOOL_START:retrieve_documents]")).toEqual({
      kind: "tool_start",
      value: "retrieve_documents",
    });
  });

  test("[TOOL_START:wikipedia_search] captures tool name", () => {
    const r = parseSSEPayload("[TOOL_START:wikipedia_search]");
    expect(r.kind).toBe("tool_start");
    expect(r.value).toBe("wikipedia_search");
  });
});

// ---------------------------------------------------------------------------
// Node start / end
// ---------------------------------------------------------------------------
describe("node_start sentinel", () => {
  test("[NODE_START:research_agent] without timestamp uses now() fallback", () => {
    const before = Date.now();
    const r = parseSSEPayload("[NODE_START:research_agent]");
    const after = Date.now();
    expect(r.kind).toBe("node_start");
    expect(r.value.node).toBe("research_agent");
    expect(r.value.startMs).toBeGreaterThanOrEqual(before);
    expect(r.value.startMs).toBeLessThanOrEqual(after);
  });

  test("[NODE_START:synthesizer|t=2024-01-01T12:00:00.000Z] parses timestamp", () => {
    const r = parseSSEPayload("[NODE_START:synthesizer|t=2024-01-01T12:00:00.000Z]");
    expect(r.kind).toBe("node_start");
    expect(r.value.node).toBe("synthesizer");
    expect(r.value.startMs).toBe(Date.parse("2024-01-01T12:00:00.000Z"));
  });

  test("[NODE_START:chat_agent|t=invalid-date] falls back to now()", () => {
    const before = Date.now();
    const r = parseSSEPayload("[NODE_START:chat_agent|t=invalid-date]");
    const after = Date.now();
    expect(r.kind).toBe("node_start");
    expect(r.value.node).toBe("chat_agent");
    expect(r.value.startMs).toBeGreaterThanOrEqual(before);
    expect(r.value.startMs).toBeLessThanOrEqual(after);
  });
});

describe("node_end sentinel", () => {
  test("[NODE_END:research_agent] returns kind:node_end", () => {
    expect(parseSSEPayload("[NODE_END:research_agent]")).toEqual({
      kind: "node_end",
      value: "research_agent",
    });
  });

  test("[NODE_END:blog_writer] captures node name", () => {
    const r = parseSSEPayload("[NODE_END:blog_writer]");
    expect(r.kind).toBe("node_end");
    expect(r.value).toBe("blog_writer");
  });
});

// ---------------------------------------------------------------------------
// Token passthrough
// ---------------------------------------------------------------------------
describe("token passthrough", () => {
  test("plain text returns kind:token", () => {
    expect(parseSSEPayload("Hello, world!")).toEqual({
      kind: "token",
      value: "Hello, world!",
    });
  });

  test("markdown heading passthrough", () => {
    const r = parseSSEPayload("## Section heading");
    expect(r.kind).toBe("token");
    expect(r.value).toBe("## Section heading");
  });

  test("empty string returns kind:token with empty value", () => {
    // empty payload.trim() = '' which matches no sentinel
    const r = parseSSEPayload("");
    expect(r.kind).toBe("token");
    expect(r.value).toBe("");
  });

  test("whitespace-only payload returns kind:token", () => {
    const r = parseSSEPayload("   ");
    // trim() = '' — falls through as token
    expect(r.kind).toBe("token");
  });

  test("multiline token rejoined via caller — individual line passthrough", () => {
    // sseParser sees one rawPayload per SSE event; multiline rejoining is done
    // by useSSE before calling parseSSEPayload
    const r = parseSSEPayload("first line\nsecond line");
    expect(r.kind).toBe("token");
    expect(r.value).toContain("first line");
  });

  test("partial sentinel lookalike is a token", () => {
    // Starts with [ but doesn't match any known pattern — falls through
    expect(parseSSEPayload("[UNKNOWN_SENTINEL]")).toEqual({
      kind: "token",
      value: "[UNKNOWN_SENTINEL]",
    });
  });

  test("sentinel prefix without closing bracket is a token", () => {
    expect(parseSSEPayload("[DONE")).toEqual({ kind: "token", value: "[DONE" });
  });

  test("sentence containing [DONE] literally is a token (model output)", () => {
    // The model might say "Sending [DONE] signal." — only exact match triggers
    const payload = "Sending [DONE] signal.";
    const r = parseSSEPayload(payload);
    // trim() !== '[DONE]' so this should fall through as token
    expect(r.kind).toBe("token");
    expect(r.value).toBe(payload);
  });
});

// ---------------------------------------------------------------------------
// Trim behaviour
// ---------------------------------------------------------------------------
describe("trim normalisation", () => {
  test("leading/trailing whitespace stripped for sentinel matching", () => {
    expect(parseSSEPayload("  [INTERRUPT]  ")).toEqual({ kind: "interrupt" });
  });

  test("but rawPayload (not trimmed) is returned as token value", () => {
    // token value should be rawPayload, not trimmed version
    const r = parseSSEPayload("  hello  ");
    expect(r.kind).toBe("token");
    expect(r.value).toBe("  hello  ");
  });
});
