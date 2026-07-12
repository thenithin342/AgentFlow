import { useState, useEffect, memo } from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import SourceChips from './SourceChips';

const AGENT_COLORS = {
  router: "var(--af-router)",
  research_agent: "var(--af-research)",
  analysis_agent: "var(--af-analysis)",
  chat_agent: "var(--af-chat)",
  synthesizer: "var(--af-synthesizer)",
  human_review: "var(--af-review)",
  blog_writer: "var(--af-blog)",
  memory_reader: "var(--af-memory)",
  memory_writer: "var(--af-memory)",
  stm_compressor: "var(--af-memory)",
};

function AgentAvatar({ agent }) {
  const icon = agent === "router" ? "🚦" :
               agent === "research_agent" ? "🔍" :
               agent === "analysis_agent" ? "📊" :
               agent === "chat_agent" ? "💬" :
               agent === "synthesizer" ? "✍️" :
               agent === "human_review" ? "🛑" : "🤖";
  const bg = AGENT_COLORS[agent] || "var(--af-router)";
  return (
    <div style={{
      width: 20, height: 20, borderRadius: "50%", background: bg, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "var(--af-bg-panel)", flexShrink: 0
    }}>
      {icon}
    </div>
  );
}

function CopyButton({ text, style }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        } catch {
          setCopied(false);
        }
      }}
      title="Copy"
      style={{
        background: "transparent",
        border: "none",
        color: copied ? "var(--af-research)" : "var(--af-text-muted)",
        cursor: "pointer",
        padding: "2px 4px",
        fontFamily: "var(--af-font-sans)",
        fontSize: 12,
        ...style
      }}
    >
      {copied ? "✓ Copied" : "📋"}
    </button>
  );
}

const COLLAPSE_THRESHOLD = 1500;

const Message = memo(function Message({ msg, onApprove, onEditResend, onSubmitEdit, editingReview, editText, setEditText, onRetry, onInlineEdit }) {
  const [collapsed, setCollapsed] = useState(
    msg.role === "agent" && !msg.streaming && (msg.text?.length ?? 0) > COLLAPSE_THRESHOLD
  );
  const [isTyping, setIsTyping] = useState(false);
  const [timeLeft, setTimeLeft] = useState(60);

  useEffect(() => {
    if (!msg.streaming) return;
    setIsTyping(true);
    const t = setTimeout(() => setIsTyping(false), 300);
    return () => clearTimeout(t);
  }, [msg.text, msg.streaming]);

  useEffect(() => {
    if (msg.role !== "review") return;
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [msg.role]);

  // Auto-uncollapse once streaming finishes
  useEffect(() => {
    if (!msg.streaming && (msg.text?.length ?? 0) <= COLLAPSE_THRESHOLD) {
      setCollapsed(false);
    }
  }, [msg.streaming, msg.text]);

  if (msg.role === "user") {
    return (
      <div
        style={{
          alignSelf: "flex-end",
          maxWidth: "78%",
          background: "var(--af-bg-surface)",
          borderRadius: 8,
          padding: "9px 13px",
          color: "var(--af-text-primary)",
          fontSize: 13.5,
          lineHeight: 1.5,
          whiteSpace: "pre-wrap",
          position: "relative",
        }}
        className="af-user-message af-msg-enter"
      >
        <div style={{ position: "relative" }}>
          {msg.text}
          {onInlineEdit && (
            <button
              className="af-inline-edit-btn"
              onClick={() => onInlineEdit(msg.text)}
              title="Edit text in input box"
            >
              ✏️
            </button>
          )}
        </div>
        {msg.timestamp && (
          <div className="af-msg-timestamp" style={{ fontSize: 10, color: "var(--af-text-muted)", marginTop: 4, textAlign: "right", fontFamily: "var(--af-font-mono)" }}>
            {msg.timestamp}
          </div>
        )}
      </div>
    );
  }

  if (msg.role === "review") {
    if (editingReview) {
      return (
        <div
          style={{
            maxWidth: "92%",
            background: "var(--af-review-bg)",
            border: "1px solid var(--af-review-border)",
            borderRadius: 8,
            padding: "12px 14px",
          }}
        >
          <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-review)", marginBottom: 7 }}>
            HUMAN_REVIEW · editing
          </div>
          <textarea
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--af-review)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--af-review-border)";
            }}
            rows={4}
            style={{
              width: "100%",
              background: "var(--af-bg-app)",
              border: "1px solid var(--af-review-border)",
              borderRadius: 5,
              color: "var(--af-text-body)",
              fontFamily: "var(--af-font-sans)",
              fontSize: 13,
              padding: 8,
              outline: "none",
              resize: "vertical",
              marginBottom: 8,
            }}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={onSubmitEdit}
              style={{
                fontFamily: "var(--af-font-mono)",
                fontSize: 11.5,
                color: "var(--af-bg-app)",
                background: "var(--af-review)",
                border: "none",
                borderRadius: 5,
                padding: "6px 12px",
                cursor: "pointer",
              }}
            >
              Submit
            </button>
          </div>
        </div>
      );
    }
    return (
      <div
        style={{
          maxWidth: "92%",
          background: "var(--af-review-bg)",
          border: "1px solid var(--af-review-border)",
          borderRadius: 8,
          padding: "12px 14px",
        }}
      >
        <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color: "var(--af-review)", marginBottom: 7, display: "flex", justifyContent: "space-between" }}>
          <span>HUMAN_REVIEW · awaiting approval</span>
          <span style={{ opacity: 0.7 }}>{timeLeft}s</span>
        </div>
        <div style={{ color: "var(--af-text-body)", fontSize: 13.5, lineHeight: 1.6, marginBottom: 11 }}>
          {msg.text}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={onApprove}
            disabled={timeLeft === 0}
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 11.5,
              color: "var(--af-bg-app)",
              background: "var(--af-review)",
              border: "none",
              borderRadius: 5,
              padding: "6px 12px",
              cursor: timeLeft === 0 ? "not-allowed" : "pointer",
              opacity: timeLeft === 0 ? 0.5 : 1,
            }}
          >
            Approve
          </button>
          <button
            onClick={() => onEditResend(msg.text)}
            disabled={timeLeft === 0}
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 11.5,
              color: "var(--af-text-body)",
              background: "transparent",
              border: "1px solid var(--af-review-border)",
              borderRadius: 5,
              padding: "6px 12px",
              cursor: timeLeft === 0 ? "not-allowed" : "pointer",
              opacity: timeLeft === 0 ? 0.5 : 1,
            }}
          >
            Edit and resend
          </button>
        </div>
      </div>
    );
  }

  // role === "agent"
  const color = (msg.error || msg.aborted) ? "var(--af-error)" : (AGENT_COLORS[msg.agent] || "var(--af-router)");
  // Phase 2: citation chips. Parsed once per text change via useMemo in
  // the parent (App) and passed in as `citations` to keep this memoized
  // component pure. Hidden while streaming so chips don't pop in mid-
  // stream before the LLM has finished writing.
  const citations = msg.citations ?? [];
  return (
    <div className="af-msg-enter" style={{ maxWidth: "84%", borderLeft: `2px solid ${color}`, padding: "6px 0 6px 12px" }}>
      <div style={{ fontFamily: "var(--af-font-mono)", fontSize: 10.5, color, marginBottom: 5, display: "flex", alignItems: "center", gap: 8 }}>
        <AgentAvatar agent={msg.agent} />
        <span>
          <span style={{ fontWeight: 500, letterSpacing: "0.04em" }}>{msg.agent ? msg.agent.toUpperCase().replace("_AGENT", "") : "AGENT"}</span> · {msg.error ? "error" : msg.aborted ? "aborted" : msg.meta}
        </span>
        {msg.text && !msg.streaming && !msg.error && !msg.aborted && (
          <CopyButton text={msg.text} />
        )}
        {msg.timestamp && (
          <span className="af-msg-timestamp" style={{ color: "var(--af-text-muted)", marginLeft: "auto" }}>{msg.timestamp}</span>
        )}
      </div>
      <div style={{ color: "var(--af-text-body)", fontSize: 13.5, lineHeight: 1.6 }}>
        {msg.error ? (
          <div className="af-error-panel" role="alert">
            <div className="af-error-panel-text">{msg.text || "An error occurred."}</div>
            {onRetry ? (
              <button
                className="af-error-panel-retry"
                onClick={() => onRetry(msg.id)}
              >
                Retry
              </button>
            ) : null}
          </div>
        ) : msg.aborted ? (
          <div className="af-aborted">[aborted] — generation was cancelled.</div>
        ) : (
          <>
            {msg.text ? (
              <>
                <div className={`markdown-body${collapsed ? " af-msg-collapsed" : ""}`}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeSanitize]}
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || "");
                        const lang = match ? match[1] : null;
                        const codeText = String(children).replace(/\n$/, "");
                        return !inline && match ? (
                          <div className="af-code-block">
                            <div className="af-code-header">
                              <span>{lang}</span>
                              <CopyButton text={codeText} />
                            </div>
                            <SyntaxHighlighter
                              style={vscDarkPlus}
                              language={lang}
                              PreTag="div"
                              customStyle={{ margin: 0, padding: "12px", background: "var(--af-bg-panel)", fontSize: "12px" }}
                            >
                              {codeText}
                            </SyntaxHighlighter>
                          </div>
                        ) : (
                          <code className={className} {...props}>{children}</code>
                        );
                      }
                    }}
                  >
                    {msg.text}
                  </ReactMarkdown>
                </div>
                {(msg.text?.length ?? 0) > COLLAPSE_THRESHOLD && !msg.streaming && (
                  <button
                    className="af-msg-expand-btn"
                    onClick={() => setCollapsed((c) => !c)}
                  >
                    {collapsed ? `▼ Show full response (${msg.text.length} chars)` : "▲ Collapse"}
                  </button>
                )}
              </>
            ) : (msg.streaming && (
              <div className="af-typing-indicator">
                <span />
                <span />
                <span />
              </div>
            ))}
            {msg.streaming ? <span className={`af-cursor ${isTyping ? "af-cursor-typing" : ""}`} aria-hidden="true" /> : null}
          </>
        )}
      </div>
      {!msg.error && !msg.aborted && <SourceChips citations={citations} />}
    </div>
  );
});

export default Message;
