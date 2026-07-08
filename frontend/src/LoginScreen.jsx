import { useState } from "react";
import { setToken, decodeToken } from "./auth";

/**
 * Login screen for AgentFlow.
 *
 * Calls POST /auth/login, stores the JWT in localStorage, then reloads
 * so the rest of the app picks up the authed state. Reload is the
 * simplest way to reset all in-memory refs (threadId, messages, trace)
 * that were initialised under the unauthed identity.
 *
 * The /auth/login endpoint is on the backend's PUBLIC_PATHS allowlist
 * (backend/main.py:280) — no Authorization header is sent here even if
 * a stale token exists in localStorage.
 *
 * Why not /auth/register: the user store is admin-provisioned. New
 * accounts are created server-side; the frontend only ever logs in.
 */
export default function LoginScreen({ onSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username.trim(),
          password,
        }),
      });
      if (!res.ok) {
        // 401 = wrong creds, 400 = malformed input. The backend
        // collapses both into a single "invalid credentials" string so
        // we don't leak which field was wrong.
        let detail = `login failed (${res.status})`;
        try {
          const body = await res.json();
          if (body?.detail) detail = body.detail;
        } catch {
          /* non-JSON body */
        }
        setError(detail);
        return;
      }
      const data = await res.json();
      if (!data?.access_token) {
        setError("server returned no token");
        return;
      }
      setToken(data.access_token);
      // Sanity: make sure the payload is decodable before we commit.
      const payload = decodeToken(data.access_token);
      if (!payload?.sub) {
        setError("server returned a malformed token");
        return;
      }
      onSuccess?.(data.access_token, payload.sub);
    } catch (err) {
      setError(err?.message || "network error");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      style={{
        minHeight: "100dvh",
        background: "var(--af-bg-app)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--af-font-sans)",
        color: "var(--af-text-primary)",
        padding: 20,
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          width: "100%",
          maxWidth: 360,
          background: "var(--af-bg-panel)",
          border: "1px solid var(--af-border)",
          borderRadius: 8,
          padding: "28px 26px 22px",
          display: "flex",
          flexDirection: "column",
          gap: 16,
          boxShadow: "0 8px 24px rgba(0,0,0,0.18)",
        }}
      >
        <div
          style={{
            fontFamily: "var(--af-font-mono)",
            fontSize: 12,
            letterSpacing: "0.08em",
            color: "var(--af-synthesizer)",
            textAlign: "center",
          }}
        >
          AGENTFLOW
        </div>
        <div
          style={{
            fontFamily: "var(--af-font-sans)",
            fontSize: 18,
            fontWeight: 500,
            textAlign: "center",
            color: "var(--af-text-primary)",
            marginTop: -4,
          }}
        >
          Sign in
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <label
            htmlFor="login-username"
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 10.5,
              color: "var(--af-text-muted)",
              letterSpacing: "0.04em",
            }}
          >
            USERNAME
          </label>
          <input
            id="login-username"
            type="text"
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={submitting}
            style={{
              background: "var(--af-bg-app)",
              border: "1px solid var(--af-border)",
              borderRadius: 5,
              padding: "9px 11px",
              color: "var(--af-text-primary)",
              fontFamily: "var(--af-font-sans)",
              fontSize: 13.5,
              outline: "none",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--af-focus)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--af-border)";
            }}
          />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <label
            htmlFor="login-password"
            style={{
              fontFamily: "var(--af-font-mono)",
              fontSize: 10.5,
              color: "var(--af-text-muted)",
              letterSpacing: "0.04em",
            }}
          >
            PASSWORD
          </label>
          <input
            id="login-password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
            style={{
              background: "var(--af-bg-app)",
              border: "1px solid var(--af-border)",
              borderRadius: 5,
              padding: "9px 11px",
              color: "var(--af-text-primary)",
              fontFamily: "var(--af-font-sans)",
              fontSize: 13.5,
              outline: "none",
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = "var(--af-focus)";
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = "var(--af-border)";
            }}
          />
        </div>
        {error && (
          <div
            role="alert"
            style={{
              background: "var(--af-error-bg)",
              border: "1px solid var(--af-status-error)",
              color: "var(--af-status-error)",
              fontFamily: "var(--af-font-sans)",
              fontSize: 12.5,
              borderRadius: 5,
              padding: "8px 10px",
            }}
          >
            {error}
          </div>
        )}
        <button
          type="submit"
          disabled={submitting || !username.trim() || !password}
          style={{
            background: "var(--af-synthesizer)",
            color: "var(--af-bg-app)",
            border: "none",
            borderRadius: 5,
            padding: "10px 14px",
            fontFamily: "var(--af-font-mono)",
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: "0.06em",
            cursor: submitting || !username.trim() || !password ? "not-allowed" : "pointer",
            opacity: submitting || !username.trim() || !password ? 0.55 : 1,
            marginTop: 2,
          }}
        >
          {submitting ? "SIGNING IN…" : "SIGN IN"}
        </button>
        <div
          style={{
            fontFamily: "var(--af-font-mono)",
            fontSize: 10,
            color: "var(--af-text-muted)",
            textAlign: "center",
            marginTop: 2,
            lineHeight: 1.5,
          }}
        >
          Accounts are provisioned by the server admin.
        </div>
      </form>
    </div>
  );
}
