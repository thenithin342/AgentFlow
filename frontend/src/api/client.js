import { getToken, clearToken } from "../auth";

const API_BASE = import.meta.env.VITE_API_BASE || "";
export const apiUrl = (path) => `${API_BASE}${path}`;

export async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token && !headers.Authorization) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(apiUrl(path), { ...options, headers });
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event("agentflow:auth_error"));
  }
  return res;
}
