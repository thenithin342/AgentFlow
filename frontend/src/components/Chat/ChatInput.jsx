import React, { useRef } from "react";
import { MAX_MESSAGE_CHARS } from "../../constants";

const SEND_BUTTON_BASE_STYLE = {
  background: "transparent",
  border: "none",
  color: "var(--af-synthesizer)",
  fontSize: 18,
  lineHeight: 1,
  padding: 6,
};

const INPUT_STYLE_BASE = {
  flex: 1,
  background: "var(--af-bg-panel)",
  border: "1px solid var(--af-border)",
  borderRadius: 6,
  padding: "8px 12px",
  color: "var(--af-text-primary)",
  fontFamily: "var(--af-font-sans)",
  fontSize: 13,
  outline: "none",
};

export default function ChatInput({
  input,
  setInput,
  isStreaming,
  isUploading,
  reviewRequired,
  setReviewRequired,
  handleSend,
  handleFileChange,
  inputRef,
}) {
  const fileInputRef = useRef(null);
  
  const autoGrow = (el) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  const sendDisabled = input.trim().length === 0 || isStreaming || isUploading || input.length > MAX_MESSAGE_CHARS;

  return (
    <div
      style={{
        flexShrink: 0,
        borderTop: "1px solid var(--af-border)",
        padding: "10px 14px",
        display: "flex",
        gap: 8,
        alignItems: "center",
      }}
    >
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isStreaming || isUploading}
        aria-label="Attach PDF"
        style={{
          background: "transparent",
          border: "none",
          color: "var(--af-text-muted)",
          cursor: (isStreaming || isUploading) ? "not-allowed" : "pointer",
          fontSize: 18,
          lineHeight: 1,
          padding: 6,
          opacity: (isStreaming || isUploading) ? 0.5 : 1,
        }}
      >
        {isUploading ? "⏳" : "📎"}
      </button>
      <button
        onClick={() => setReviewRequired((r) => !r)}
        disabled={isStreaming}
        aria-label="Toggle review mode"
        title={reviewRequired ? "Review mode ON — response will pause for approval" : "Review mode OFF"}
        style={{
          background: reviewRequired ? "rgba(227,179,86,0.15)" : "transparent",
          border: reviewRequired ? "1px solid var(--af-review-border)" : "1px solid transparent",
          borderRadius: 5,
          color: reviewRequired ? "var(--af-review)" : "var(--af-text-muted)",
          cursor: isStreaming ? "not-allowed" : "pointer",
          fontFamily: "var(--af-font-mono)",
          fontSize: 10,
          letterSpacing: "0.04em",
          padding: "4px 7px",
          opacity: isStreaming ? 0.5 : 1,
          transition: "all 0.15s",
        }}
      >
        REVIEW
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        aria-label="Upload PDF document"
        onChange={handleFileChange}
        style={{ display: "none" }}
        data-testid="file-upload"
      />
      <textarea
        id="chat-input"
        ref={inputRef}
        value={input}
        aria-label="Chat message"
        rows={1}
        readOnly={isStreaming}
        aria-busy={isStreaming}
        onChange={(e) => {
          setInput(e.target.value);
          autoGrow(e.currentTarget);
        }}
        onKeyDown={(e) => {
          if (e.key !== "Enter") return;
          if (e.nativeEvent.isComposing || e.keyCode === 229) return;
          if (e.shiftKey) return;
          e.preventDefault();
          handleSend();
        }}
        placeholder="Ask agentflow..."
        style={INPUT_STYLE_BASE}
      />
      {input.trim().length > 0 && (
        <span className="af-word-count">
          {input.trim().split(/\s+/).length} words
          {input.length > MAX_MESSAGE_CHARS * 0.8 && (
            <span style={{ marginLeft: 8, color: input.length > MAX_MESSAGE_CHARS ? "var(--af-error)" : "inherit" }}>
              | {input.length} / {MAX_MESSAGE_CHARS} chars
            </span>
          )}
        </span>
      )}
      <button
        onClick={handleSend}
        disabled={sendDisabled}
        aria-label="Send message"
        style={{
          ...SEND_BUTTON_BASE_STYLE,
          cursor: sendDisabled ? "not-allowed" : "pointer",
          opacity: sendDisabled ? 0.5 : 1,
        }}
      >
        ↑
      </button>
    </div>
  );
}
