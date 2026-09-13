import { getToken, clearToken, setToken } from "../auth";

const API_BASE = import.meta.env.VITE_API_BASE || "";
export const apiUrl = (path) => `${API_BASE}${path}`;

// Serialise concurrent 401 refresh attempts: if multiple requests get a 401
// simultaneously they would each call silentRefresh(), race to store the new
// token, and likely invalidate each other. One shared in-flight Promise means
// only one refresh hits the network; all waiting callers get the same result.
let _refreshPromise = null;

export async function silentRefresh() {
  if (_refreshPromise) return _refreshPromise;
  _refreshPromise = (async () => {
    const token = getToken();
    if (!token) return false;
    try {
      const refreshRes = await fetch(apiUrl("/auth/refresh"), {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(10_000),
      });
      if (refreshRes.ok) {
        const data = await refreshRes.json();
        setToken(data.access_token);
        return data.access_token;
      }
    } catch (e) {
      console.error("Token refresh failed:", e);
    }
    return false;
  })();
  try {
    return await _refreshPromise;
  } finally {
    _refreshPromise = null;
  }
}

// Default request timeout (ms). Keeps inflight fetches from hanging
// indefinitely on a stalled connection. SSE /chat uses its own reader
// with a watchdog; this covers all other routes.
const DEFAULT_TIMEOUT_MS = 25_000;

// Upload timeout: the backend must load the embedding model (~80 MB),
// run FastEmbed over all chunks, and write the FAISS index to disk.
// On a cold Render free-tier instance this can take 60-90 s.
const UPLOAD_TIMEOUT_MS = 120_000;

export async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }

  // Compose caller-provided signal with a hard per-request deadline.
  // AbortSignal.any() is available in all evergreen browsers (2024+).
  // Use a longer timeout for /upload because embedding can take 60-90 s on cold start.
  const isUpload = path === "/upload";
  const timeoutMs = isUpload ? UPLOAD_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeoutSignal])
    : timeoutSignal;

  let res = await fetch(apiUrl(path), { ...options, headers, signal });

  if (res.status === 401) {
    // Attempt silent renewal if we have a token
    const newToken = await silentRefresh();
    if (newToken) {
      // Retry the original request with the new token
      headers.Authorization = `Bearer ${newToken}`;
      res = await fetch(apiUrl(path), { ...options, headers, signal });
      return res;
    }

    // If refresh failed or there was no token, trigger logout
    clearToken();
    window.dispatchEvent(new Event("agentflow:auth_error"));
  }

  return res;
}
