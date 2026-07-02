/** Shared limits — keep in sync with backend/constants.py */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
export const MAX_MESSAGE_CHARS = 16_000;

export const TRACE_STREAM_NODES = new Set([
  "router",
  "chat_agent",
  "research_agent",
  "analysis_agent",
  "synthesizer",
]);
