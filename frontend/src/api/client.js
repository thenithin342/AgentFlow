import { getToken, clearToken } from "../auth";

const API_BASE = import.meta.env.VITE_API_BASE || "";
export const apiUrl = (path) => `${API_BASE}${path}`;

export async function silentRefresh() {
  const token = getToken();
  if (!token) return false;
  try {
    const refreshRes = await fetch(apiUrl("/auth/refresh"), {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` }
    });
    if (refreshRes.ok) {
      const data = await refreshRes.json();
      const { setToken } = await import("../auth");
      setToken(data.access_token);
      return data.access_token;
    }
  } catch (e) {
    console.error("Token refresh failed:", e);
  }
  return false;
}

export async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }
  let res = await fetch(apiUrl(path), { ...options, headers });
  
  if (res.status === 401) {
    // Attempt silent renewal if we have a token
    const newToken = await silentRefresh();
    if (newToken) {
      // Retry the original request with the new token
      headers.Authorization = `Bearer ${newToken}`;
      res = await fetch(apiUrl(path), { ...options, headers });
      return res;
    }
    
    // If refresh failed or there was no token, trigger logout
    clearToken();
    window.dispatchEvent(new Event("agentflow:auth_error"));
  }
  
  return res;
}
