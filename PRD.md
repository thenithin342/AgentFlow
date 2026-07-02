# AgentFlow — Product Requirements Document

**Version:** 1.0
**Owner:** Nithin
**Status:** Draft — active development

---

## 1. Summary

AgentFlow is a multi-agent knowledge assistant built on LangGraph. A user submits a query through a chat interface; a router node classifies intent and dispatches the query to one specialized agent (Research, Analysis, or Chat); the agent's output is rewritten by a synthesizer node; an optional human-review checkpoint allows a draft response to be approved or edited before it is returned. Conversation state is durably persisted, so sessions survive backend restarts and users can resume conversations by `thread_id`.

The project exists to demonstrate production-grade agentic system design — stateful graph orchestration, conditional routing, tool-augmented reasoning, durable checkpointing, and human-in-the-loop control — as a portfolio piece.

---

## 2. Problem Statement

Most "chatbot" portfolio projects are a single LLM call wrapped in a UI. They don't demonstrate the skills that distinguish an AI engineer who can build production agentic systems: routing logic, multi-agent coordination, persistent state, and safety checkpoints. AgentFlow is scoped specifically to exercise each of these in a way that is small enough to finish in 4–5 weeks but deep enough to discuss in technical interviews.

---

## 3. Goals

- Ship a working multi-agent system with at least three distinct, independently testable agent nodes.
- Demonstrate conditional routing driven by an LLM classification step, not hardcoded keyword matching.
- Persist conversation state across process restarts using a real checkpointer (SQLite via `SqliteSaver` or `PostgresSaver`).
- Implement at least one genuine human-in-the-loop interrupt (not a simulated one).
- Support retrieval-augmented generation over user-uploaded documents.
- Stream responses to the frontend token-by-token.
- Produce a polished GitHub repo: README with architecture diagram, setup instructions, and a recorded demo.

### Non-goals

- Multi-tenant auth, billing, or production deployment infrastructure.
- Support for more than a handful of concurrent users (this is a portfolio app, not a SaaS product).
- Fine-tuning or training any model — all models are used via free inference APIs or local Ollama.
- Voice interface or mobile app.

---

## 4. Users and Use Cases

The primary "user" is a recruiter, interviewer, or hiring manager evaluating the project, plus Nithin himself as the day-to-day tester during development. The product must therefore be easy to run locally in under five minutes and easy to demo live.

**Use case 1 — Research question.** User asks a question requiring current information ("what are the latest developments in X"). Router sends this to the Research agent, which calls a web search tool, retrieves and cites sources, and returns a grounded answer.

**Use case 2 — Document Q&A.** User uploads a PDF, then asks a question whose answer requires that document's content. The Research agent's retrieval tool pulls from the FAISS index built from the uploaded file.

**Use case 3 — Reasoning / analysis task.** User asks for a comparison, summary, or computation (e.g. "summarize and compare section 3 vs section 5"). Router sends this to the Analysis agent.

**Use case 4 — Casual conversation.** User sends a follow-up like "thanks, can you shorten that?" with no need for tools. Router sends this to the Chat agent, which responds directly without any tool calls, keeping latency low.

**Use case 5 — Human review.** When review mode is enabled, the synthesized draft pauses for explicit human approval or edits before being returned to the user.

---

## 5. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Router node classifies each incoming query into one of: research, analysis, chat | P0 |
| FR2 | Research agent can call a web search tool (Tavily) and a retrieval tool (FAISS) | P0 |
| FR3 | Analysis agent can call a code/calculation tool and reason over retrieved document chunks | P0 |
| FR4 | Chat agent responds directly with no tool calls for conversational turns | P0 |
| FR5 | Synthesizer merges agent output(s) into one coherent final response | P0 |
| FR6 | Conversation state persists across backend restarts, keyed by `thread_id` | P0 |
| FR7 | Human-in-the-loop interrupt pauses graph execution after synthesis when review mode is on | P0 |
| FR8 | Users can upload a PDF; it is chunked, embedded, and indexed into FAISS for retrieval | P0 |
| FR9 | Backend streams tokens to the frontend as they're generated | P1 |
| FR10 | Frontend displays which agent(s) handled a given query (for demo transparency) | P1 |
| FR11 | Conversation history is viewable and resumable from a thread list | P2 |
| FR12 | Basic rate-limit/retry handling around the LLM API calls | P2 |

---

## 6. Non-Functional Requirements

Latency should stay under ~5 seconds for chat-only queries and under ~15 seconds for research queries involving a web search call, acceptable for a demo context rather than a production SLA. The system must run entirely on free-tier APIs (Groq, Tavily, Google AI Studio) plus local compute, with no required paid services. Code should be modular enough that each agent node, the router, and the synthesizer can be unit-tested independently of the full graph. The repo must be runnable from a clean clone with a documented `.env` setup in under five minutes.

---

## 7. Success Metrics

Since this is a portfolio project rather than a live product, success is measured by: (1) all P0 functional requirements working end-to-end in a live demo without crashing, (2) a router classification accuracy that "feels" correct in informal testing across at least 20 varied test queries, (3) a GitHub repo that a stranger could clone and run unassisted, and (4) the ability to clearly explain every architectural decision (why a router, why a checkpointer, why an interrupt) in an interview setting.

---

## 8. Risks and Mitigations

Free-tier API rate limits (especially Groq's 6k TPM cap on 70b models) could interrupt a live demo — mitigated by using the faster, higher-limit `llama-3.1-8b-instant` for agent nodes and reserving the 70b model only for router/synthesizer calls. Router misclassification could send queries to the wrong agent — mitigated by few-shot examples in the router prompt and a fallback path that defaults to the Chat agent when classification confidence is low. Scope creep is a real risk given the 8-phase roadmap — mitigated by treating Phases 1–6 as the P0 cutline and Phases 7–8 (RAG, full polish) as stretch goals if time runs short.

---

## 9. Milestones

Phase 1–2 (routing skeleton) by end of week 1. Phase 3–4 (tools + persistence) by end of week 2. Phase 5–6 (multi-agent + human-in-the-loop) by end of week 3. Phase 7–8 (RAG + full-stack polish) by end of week 5. See `TECH_STACK.md` and `DESIGN_DOC.md` for implementation detail.
