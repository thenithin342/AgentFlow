"""Shared limits and trace configuration for API + graph streaming."""

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_MESSAGE_CHARS = 16_000

# Nodes surfaced in SSE trace events and the frontend execution rail.
TRACE_STREAM_NODES = frozenset({
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
})

# Nodes whose on_chat_model_stream tokens are forwarded to the client.
SSE_TOKEN_NODES = frozenset({
    "synthesizer",
    "chat_agent",
    "research_agent",
    "analysis_agent",
    "blog_writer",
})
