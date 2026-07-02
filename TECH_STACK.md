# AgentFlow — Tech Stack

**Companion to:** `PRD.md`, `DESIGN_DOC.md`

---

## 1. Orchestration

**LangGraph** (`langgraph>=0.2`) is the core framework — `StateGraph`, conditional edges, checkpointers, and the `interrupt` API are what this entire project is built to demonstrate. **LangChain** (`langchain`, `langchain-core`) supplies the tool-calling abstractions (`@tool`, `create_react_agent`), message types, and the FAISS/Tavily integration wrappers that sit underneath LangGraph nodes.

## 2. LLM Providers (free tier)

**Groq** (`langchain-groq`) is primary: `llama-3.3-70b-versatile` for the Router and Synthesizer (needs real reasoning), `llama-3.1-8b-instant` for the Research, Analysis, and Chat agent nodes (fast, reliable tool calling, much higher free-tier rate limit at 30k TPM vs. 6k TPM for the 70b model). **Google AI Studio** (`langchain-google-genai`, `gemini-2.0-flash`) is the fallback if Groq's rate limits are hit during heavy testing — generous 1M tokens/day free allowance and equally strong tool calling. Local **Ollama** running `Qwen3-8B` is an optional fully-offline path for development without burning API quota, given Nithin's existing local setup (Intel Iris Xe iGPU, 16GB RAM).

## 3. Tools

**Tavily** (`tavily-python` / `langchain-community`'s `TavilySearchResults`) provides the web search tool for the Research agent — purpose-built for LLM agents, with a free tier of 1,000 searches/month. A sandboxed calculator tool — a hand-rolled AST-walked expression evaluator that rejects names, calls, and attribute access, and caps expression length, AST depth, and exponent magnitudes — handles the Analysis agent's computation needs without the security and reliability pitfalls of `PythonREPLTool`.

## 4. Retrieval / RAG

**FAISS** (`faiss-cpu`) is the vector store — local, free, no hosted service required, and fast enough for single-user/demo-scale document collections. Embeddings come from a local `sentence-transformers` model (e.g. `all-MiniLM-L6-v2` via `langchain-huggingface`) to avoid burning API quota on embedding calls, which tend to be high-volume relative to chat completions.

## 5. Persistence

**SQLite** via LangGraph's `SqliteSaver` checkpointer for conversation state — zero setup, file-based, perfect for a portfolio project. The design doc notes `PostgresSaver` as a documented upgrade path (Nithin has prior experience wiring `PostgresSaver` for durable checkpointing) if the project later needs concurrent multi-user writes.

## 6. Backend

**FastAPI** serves the chat, upload, and review endpoints, with `StreamingResponse` wrapping LangGraph's `astream_events` for token-level streaming to the frontend. **Uvicorn** as the ASGI server.

## 7. Frontend

**React** (via Vite) + **Tailwind CSS** for a minimal, fast-to-build chat interface — consistent with Nithin's existing stack experience from BookHub and other projects. No heavyweight state library needed; React's built-in state plus the Fetch API's streaming reader is sufficient at this scale.

## 8. Dev Tooling

**Python 3.11+**, `pip`/`venv` for environment management, `python-dotenv` for API key management via a `.env` file (never committed — `.env.example` ships instead), and `pytest` for the node-level and graph-level tests described in the design doc.

## 9. Summary Table

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph + LangChain | Core subject of the project |
| Smart LLM | Groq `llama-3.3-70b-versatile` | Router + Synthesizer reasoning |
| Fast LLM | Groq `llama-3.1-8b-instant` | Agent tool-calling, high rate limit |
| Fallback LLM | Google Gemini 2.0 Flash | High daily token allowance |
| Local LLM (optional) | Ollama + Qwen3-8B | Offline dev, zero API cost |
| Web search | Tavily | Built for LLM agents, free tier |
| Code/calc tool | Sandboxed calculator (AST-validated expression evaluator) | Analysis agent computation |
| Vector store | FAISS (local) | Free, fast, no hosted service |
| Embeddings | sentence-transformers (local) | Avoids API quota burn |
| Checkpointing | SQLite (`SqliteSaver`) | Zero-setup durable persistence |
| Backend | FastAPI + Uvicorn | Async, streaming-friendly |
| Frontend | React + Vite + Tailwind | Fast to build, familiar stack |
| Config | python-dotenv | Standard `.env` key management |
| Testing | pytest | Node-level + graph-level tests |

## 10. Cost

Every component above runs on a free tier or local compute. There is no required paid service anywhere in this stack — the only constraints are the free-tier rate limits documented per provider above, all of which are generous enough for development and live demos.
