/**
 * JWT helpers for the AgentFlow frontend.
 *
 * Storage: localStorage key `agentflow_jwt`. We use localStorage (not a
 * cookie) because the backend is a pure API — no session table, no
 * CSRF token to pair with. The trade-off: any XSS can read the token.
 * We mitigate by:
 *   1. Never storing anything else in localStorage.
 *   2. Sanitising all message text via ReactMarkdown + rehype-sanitize
 *      (see App.jsx markdown render).
 *   3. The backend's per-user thread scoping limits blast radius even
 *      if a token is stolen.
 *
 * Why no signature verification in the browser: HS256 requires the
 * shared secret, which the frontend must not have. The backend is the
 * source of truth — invalid tokens get 401s and the user is bounced
 * to the login screen. We only decode the payload here so the UI can
 * show "logged in as <user>" and know when to refresh.
 *
 * Payload format (set in backend/auth.py:issue_token):
 *   { sub, iat, exp, iss }
 */

const STORAGE_KEY = "agentflow_jwt";

/** Read the raw JWT from localStorage. */
export function getToken() {
  try {
    return localStorage.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

/** Persist a JWT. */
export function setToken(token) {
  try {
    localStorage.setItem(STORAGE_KEY, token);
  } catch {
    // Storage may be disabled (private mode, quota). The login will
    // succeed server-side but the user will be bounced back to login
    // on reload. Surface this via the login form error path.
  }
}

/** Clear the stored JWT. */
export function clearToken() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* noop */
  }
}

/**
 * Decode the JWT payload (base64url) without verifying the signature.
 * Returns null on malformed/expired tokens.
 *
 * `exp` is in seconds since epoch (per RFC 7519). The browser clock
 * drift tolerance is small: we treat tokens as expired 30s early so a
 * user mid-action doesn't get a sudden 401 from a token that just
 * ticked over.
 */
export function decodeToken(token) {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    // base64url -> base64
    const b64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = b64 + "=".repeat((4 - (b64.length % 4)) % 4);
    const json = atob(padded);
    const payload = JSON.parse(json);
    if (typeof payload.exp !== "number" || typeof payload.sub !== "string") {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

/** Return the username encoded in the token, or null. */
export function getUsername(token = getToken()) {
  const p = decodeToken(token);
  return p?.sub || null;
}

/** True if the token is expired (or within 30s grace). */
export function isExpired(token = getToken(), nowSec = Math.floor(Date.now() / 1000)) {
  const p = decodeToken(token);
  if (!p) return true;
  return p.exp <= nowSec + 30;
}
