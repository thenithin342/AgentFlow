# AgentFlow — UI Design System

**Companion to:** `PRD.md`, `DESIGN_DOC.md`, `TECH_STACK.md`

---

## 1. Design thesis

Most chatbot UIs hide the system and show only a conversation. AgentFlow's whole point is the opposite: the multi-agent routing *is* the product, so the UI should make that visible rather than decorating a generic chat window with a sparkle icon and a gradient avatar.

The signature element is the **execution trace rail** — a live, always-visible log of graph nodes as they fire (router → agent → synthesizer → review), rendered like an instrumentation panel, not a chat accessory. Every assistant message is also visually attributed to the agent that produced it, so a viewer can watch routing decisions happen in real time during a demo, which is exactly the thing worth showing an interviewer.

This deliberately avoids the three most common "AI-generated" UI defaults: warm cream + serif + terracotta, near-black + single neon accent, and dense broadsheet columns. AgentFlow's palette is a warm graphite (not pure black) with *categorical* identity colors per graph node — the color differences carry real meaning (which agent handled this), not decoration.

---

## 2. Design tokens

### Color

| Token | Hex | Role |
|---|---|---|
| `bg-app` | `#14161A` | App background |
| `bg-panel` | `#191C22` | Trace rail / panel background |
| `bg-surface` | `#21252D` | Message bubbles, input field |
| `border` | `#262A33` | Hairline dividers |
| `text-primary` | `#ECE9E2` | Primary text (warm off-white, not pure white) |
| `text-muted` | `#676C78` | Timestamps, secondary labels |
| `text-body` | `#DEDCD5` | Message body text |

**Agent identity colors** (used consistently in the trace rail dot, the message's left border, and its meta-label):

| Node | Hex | Rationale |
|---|---|---|
| Router | `#8B8F98` (neutral gray) | Structural/dispatch role, not a persona — deliberately uncolored |
| Research agent | `#4FC3B8` (teal) | Matches the "agent" family used in `architecture.svg` |
| Analysis agent | `#6FA3D6` (dusty blue) | Distinct hue within the agent family |
| Chat agent | `#9AA0AC` (slate) | Neutral — the no-tools fast path |
| Synthesizer | `#FF6B57` (coral) | Matches `architecture.svg`'s synthesizer color |
| Human review | `#E3B356` (amber/gold) | Matches `architecture.svg`'s human review color |

Do not introduce additional hues beyond this table — if a new node type is added to the graph, extend this table rather than reusing an existing color for two meanings.

### Typography

Two-face system, both from the IBM Plex superfamily (free, Google Fonts, cohesive but distinct roles):

- **IBM Plex Mono** — the "system voice." Used for the trace rail, message meta-labels (`RESEARCH_AGENT · 3 sources · 1.4s`), timestamps, thread IDs, and the input placeholder. This is what makes the UI read as an instrumentation panel rather than a chat app.
- **IBM Plex Sans** — the "content voice." Used for actual message body text and any longer-form UI copy. Readable, humanist, not the system's default sans.

Never mix these roles — if it's system metadata, it's mono; if it's something a human said or read, it's sans.

### Spacing and shape

8px base unit. Panels use `6–8px` border radius (not fully rounded — this is a technical tool, not a consumer app). Message attribution uses a `2px` solid left border in the agent's identity color, square corners (no radius on single-sided borders). Hairline dividers are `1px solid var(--border)`.

---

## 3. Layout

Two-column desktop layout: a fixed-width (~180px) trace rail on the left, main conversation column filling the rest. On narrow/mobile viewports, the trace rail collapses into a horizontal, scrollable strip above the message list, or a slide-out drawer triggered by a small icon in the top bar — never removed entirely, since it's the differentiating feature.

```
┌─────────────┬──────────────────────────────┐
│ EXECUTION   │  AGENTFLOW          thread id │
│ TRACE       ├──────────────────────────────┤
│             │                                │
│ ● router    │         [user message]         │
│ ● research  │  [agent response, colored      │
│   _agent    │   left border + mono meta]     │
│ ● synth-    │                                │
│   esizer    │         [user message]         │
│             │  [agent response...]            │
│             │                                │
│             │  [human review panel, amber]   │
│             ├──────────────────────────────┤
│             │  [input.......................] │
└─────────────┴──────────────────────────────┘
```

---

## 4. Component notes

**Trace rail entries** are timestamped, monospace, with a small colored dot matching the node's identity color. The most recently active/in-flight node's dot pulses gently (opacity animation, ~1.6s cycle) — this is the one deliberate motion moment in the UI; everything else is static.

**Assistant messages** are not bubbles — they're left-bordered panels (matching the trace rail's visual language), with a mono meta-line above the body text stating which agent answered, what it did (tool calls, source count), and how long it took. This is what lets a viewer connect a response back to the graph execution that produced it.

**User messages** are right-aligned flat panels with no border accent — visually quieter than agent responses, since the agent identity is the thing worth emphasizing.

**Human review panel** gets its own treatment: amber-tinted background (not just a border), because it represents a paused, waiting-for-input state that should read as distinct from a normal completed message. Two actions: "Approve" (filled, amber) and "Edit and resend" (outlined).

**Input bar** is flat and quiet — dark field, mono placeholder text, single accent-colored send icon (using the synthesizer's coral, since that's the "final voice" color).

---

## 5. Implementation

See `frontend/src/index.css` for the CSS custom properties and `frontend/src/App.jsx` for the working React implementation of this design, built to slot into the Phase 8 FastAPI streaming backend described in `DESIGN_DOC.md` section 7–8.
