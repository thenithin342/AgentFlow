import React, { useState, useEffect } from "react";
import { apiFetch } from "../api/client";

export default function AdminPage() {
  const [users, setUsers] = useState([]);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [passwordChangeTarget, setPasswordChangeTarget] = useState(null);
  const [newPasswordForUser, setNewPasswordForUser] = useState("");

  useEffect(() => {
    fetchUsers();
  }, []);

  async function fetchUsers() {
    try {
      const res = await apiFetch("/admin/users");
      const text = await res.text();
      if (res.ok) {
        try {
          const data = JSON.parse(text);
          setUsers(data);
        } catch {
          setError(`Server returned non-JSON response (status ${res.status}): ${text.slice(0, 200)}`);
        }
      } else {
        let detail = `HTTP ${res.status}`;
        try { detail = JSON.parse(text).detail || detail; } catch { /* ignore */ }
        setError(`Failed to fetch users — ${detail}`);
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
    }
  }

  async function handleCreateUser(e) {
    e.preventDefault();
    if (!newUsername || !newPassword) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/admin/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: newUsername, password: newPassword })
      });
      const text = await res.text();
      if (res.ok) {
        setNewUsername("");
        setNewPassword("");
        fetchUsers();
      } else {
        let detail = `HTTP ${res.status}`;
        try { detail = JSON.parse(text).detail || detail; } catch { /* ignore */ }
        setError(`Failed to create user — ${detail}`);
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteUser(username) {
    if (!window.confirm(`Are you sure you want to delete user '${username}'?`)) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(`/admin/users/${username}`, { method: "DELETE" });
      if (res.ok) {
        fetchUsers();
      } else {
        const text = await res.text();
        let detail = `HTTP ${res.status}`;
        try { detail = JSON.parse(text).detail || detail; } catch { /* ignore */ }
        setError(`Failed to delete user — ${detail}`);
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleChangePassword(username) {
    if (!newPasswordForUser) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(`/admin/users/${username}/password`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_password: newPasswordForUser })
      });
      if (res.ok) {
        setPasswordChangeTarget(null);
        setNewPasswordForUser("");
        alert(`Password for ${username} changed successfully.`);
      } else {
        const text = await res.text();
        let detail = `HTTP ${res.status}`;
        try { detail = JSON.parse(text).detail || detail; } catch { /* ignore */ }
        setError(`Failed to change password — ${detail}`);
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "40px", flex: 1, overflowY: "auto", background: "var(--af-bg-main)" }}>
      <h2>User Management (Admin)</h2>
      {error && <div style={{ color: "var(--af-research)", marginBottom: 20 }}>{error}</div>}
      
      <div style={{ background: "var(--af-bg-panel)", padding: 20, borderRadius: 8, marginBottom: 40, border: "1px solid var(--af-border)" }}>
        <h3>Create New User</h3>
        <form onSubmit={handleCreateUser} style={{ display: "flex", gap: 10, marginTop: 10 }}>
          <input
            type="text"
            placeholder="Username"
            value={newUsername}
            onChange={e => setNewUsername(e.target.value)}
            style={{ padding: "8px 12px", borderRadius: 4, border: "1px solid var(--af-border)", background: "var(--af-bg-main)", color: "var(--af-text-primary)" }}
          />
          <input
            type="password"
            placeholder="Password"
            value={newPassword}
            onChange={e => setNewPassword(e.target.value)}
            style={{ padding: "8px 12px", borderRadius: 4, border: "1px solid var(--af-border)", background: "var(--af-bg-main)", color: "var(--af-text-primary)" }}
          />
          <button type="submit" disabled={loading} style={{ background: "var(--af-synthesizer)", color: "var(--af-bg-main)", padding: "8px 16px", borderRadius: 4, border: "none", cursor: "pointer" }}>
            {loading ? "Creating..." : "Create User"}
          </button>
        </form>
      </div>

      <div style={{ background: "var(--af-bg-panel)", padding: 20, borderRadius: 8, border: "1px solid var(--af-border)" }}>
        <h3>Registered Users</h3>
        <table style={{ width: "100%", marginTop: 10, borderCollapse: "collapse", textAlign: "left" }}>
          <thead>
            <tr>
              <th style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>Username</th>
              <th style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>Created At</th>
              <th style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map(u => (
              <tr key={u.username}>
                <td style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>{u.username}</td>
                <td style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>{new Date(u.created_at * 1000).toLocaleString()}</td>
                <td style={{ padding: 10, borderBottom: "1px solid var(--af-border)" }}>
                  {passwordChangeTarget === u.username ? (
                    <div style={{ display: "flex", gap: 8 }}>
                      <input
                        type="password"
                        placeholder="New Password"
                        value={newPasswordForUser}
                        onChange={e => setNewPasswordForUser(e.target.value)}
                        style={{ padding: "4px 8px", width: 120, borderRadius: 4, border: "1px solid var(--af-border)", background: "var(--af-bg-main)", color: "var(--af-text-primary)" }}
                      />
                      <button onClick={() => handleChangePassword(u.username)} disabled={loading} style={{ background: "var(--af-synthesizer)", color: "var(--af-bg-main)", padding: "4px 8px", borderRadius: 4, border: "none", cursor: "pointer" }}>Save</button>
                      <button onClick={() => { setPasswordChangeTarget(null); setNewPasswordForUser(""); }} style={{ background: "transparent", color: "var(--af-text-muted)", border: "none", cursor: "pointer" }}>Cancel</button>
                    </div>
                  ) : (
                    <div style={{ display: "flex", gap: 10 }}>
                      <button onClick={() => setPasswordChangeTarget(u.username)} style={{ background: "transparent", color: "var(--af-text-primary)", border: "1px solid var(--af-border)", padding: "4px 8px", borderRadius: 4, cursor: "pointer" }}>Change Password</button>
                      {u.username !== "admin" && (
                        <button onClick={() => handleDeleteUser(u.username)} disabled={loading} style={{ background: "transparent", color: "var(--af-research)", border: "1px solid var(--af-research)", padding: "4px 8px", borderRadius: 4, cursor: "pointer" }}>Delete</button>
                      )}
                    </div>
                  )}
                </td>
              </tr>
            ))}
            {users.length === 0 && (
              <tr>
                <td colSpan="3" style={{ padding: 10, textAlign: "center", color: "var(--af-text-muted)" }}>No users found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
