import React, { useState } from "react";
import { formatLastSeen } from "../../utils";

export default function Sidebar({
  sidebarOpen,
  threadList,
  threadId,
  loadThread,
  deleteThread,
  handleNewThreadSafe,
  searchQuery,
  setSearchQuery,
}) {
  return (
    <div className={`af-sidebar af-scroll ${sidebarOpen ? "" : "collapsed"}`}>
      <div className="af-sidebar-header">
        <span className="af-sidebar-title">Threads</span>
        <button
          className="af-new-chat-btn"
          onClick={handleNewThreadSafe}
          title="New thread (Cmd+K)"
        >
          + New
        </button>
      </div>
      <div className="af-sidebar-search">
        <input
          type="text"
          placeholder="Search conversations…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>
      <div className="af-sidebar-threads">
        {[...threadList]
          .sort((a, b) => {
            // newest first. last_seen is ISO8601 when parseable; fall back to
            // raw checkpoint_id (UUID v6 sorts lexicographically by time).
            const ta = a.last_seen || a.checkpoint_id || a.thread_id;
            const tb = b.last_seen || b.checkpoint_id || b.thread_id;
            return tb < ta ? -1 : tb > ta ? 1 : 0;
          })
          .filter(t =>
            (t.preview || t.thread_id)
              .toLowerCase()
              .includes(searchQuery.toLowerCase())
          )
          .map((t) => (
            <div
              key={t.thread_id}
              className={`af-thread-item ${t.thread_id === threadId ? "active" : ""}`}
              onClick={() => loadThread(t.thread_id)}
            >
              <div className="af-thread-preview">
                {t.preview || t.thread_id.slice(0, 20) + "…"}
              </div>
              <div className="af-thread-meta">
                <span className="af-thread-time">
                  {formatLastSeen(t.last_seen)}
                </span>
                {t.route && (
                  <span
                    className="af-thread-route-badge"
                    style={{
                      color: t.route === "blog" ? "var(--af-blog)"
                        : t.route === "research" ? "var(--af-research)"
                        : t.route === "analysis" ? "var(--af-analysis)"
                        : "var(--af-text-muted)",
                    }}
                  >
                    {t.route}
                  </span>
                )}
                {t.turn_count != null && (
                  <span style={{ fontFamily: "var(--af-font-mono)", fontSize: 9.5, color: "var(--af-text-muted)" }}>
                    {t.turn_count}t
                  </span>
                )}
              </div>
              <button
                className="af-thread-delete-btn"
                onClick={(e) => deleteThread(t.thread_id, e)}
                title="Delete thread"
                aria-label={`Delete thread ${t.thread_id.slice(0, 8)}`}
              >
                ✕
              </button>
            </div>
          ))
        }
        {threadList.length === 0 && (
          <div style={{ padding: "14px", color: "var(--af-text-muted)", fontSize: 12 }}>
            No conversations yet.
          </div>
        )}
      </div>
    </div>
  );
}
