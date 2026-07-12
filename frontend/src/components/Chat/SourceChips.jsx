import React from "react";

export default function SourceChips({ citations }) {
  if (!citations || citations.length === 0) return null;

  return (
    <div
      style={{
        marginTop: 8,
        display: "flex",
        flexWrap: "wrap",
        gap: 6,
        alignItems: "center",
      }}
    >
      <span
        style={{
          fontFamily: "var(--af-font-mono)",
          fontSize: 10,
          color: "var(--af-text-muted)",
        }}
      >
        Sources:
      </span>
      {citations.slice(0, 8).map((c, i) => (
        <a
          key={`${c.n}-${i}`}
          className="af-source-chip"
          href={c.url}
          target="_blank"
          rel="noreferrer noopener"
        >
          [{c.n}] {c.host}
        </a>
      ))}
      {citations.length > 8 ? (
        <span
          style={{
            fontFamily: "var(--af-font-mono)",
            fontSize: 10,
            color: "var(--af-text-muted)",
          }}
        >
          …
        </span>
      ) : null}
    </div>
  );
}
