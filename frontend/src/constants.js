/** Shared limits — keep in sync with backend/constants.py */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
export const MAX_MESSAGE_CHARS = 16_000;

export const TRACE_STREAM_NODES = new Set([
  "router",
  "chat_agent",
  "research_agent",
  "analysis_agent",
  "blog_writer",
  "synthesizer",
  "human_review",
  "memory_reader",
  "memory_writer",
  "stm_compressor",
]);

export const SSE_TOKEN_NODES = new Set([
  "synthesizer",
  "chat_agent",
  "research_agent",
  "analysis_agent",
  "blog_writer",
]);
