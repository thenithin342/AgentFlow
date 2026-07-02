import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/*
 * Vite config for the AgentFlow frontend.
 *
 * The FastAPI backend runs on :8000 in dev. The dev server proxies all API
 * paths to it so the React app can call `fetch("/chat")` etc. directly.
 * The preview server (used for `npm run build` + `npm run preview`, e.g.
 * for a production-style smoke test) has the SAME proxy requirement —
 * `vite preview` does NOT honor `server.proxy`, so we set `preview.proxy`
 * explicitly. Otherwise a built bundle loaded via `vite preview` would
 * hit itself (port 4173) and 404 on every API call.
 *
 * VITE_API_BASE: optional override (e.g. when deploying the static bundle
 * behind a reverse proxy on a different host). When set, fetch() in
 * App.jsx reads it as the base URL prefix. When unset (the default for
 * dev), fetch() uses relative paths and the proxy above does the work.
 *
 * API auth: AGENTFLOW_API_KEY is read from the repo-root .env by Vite at
 * dev/preview time and injected into proxied requests server-side. It is
 * NEVER bundled into the client JS — browsers must not hold bearer tokens
 * for a public static bundle. Production: terminate auth at nginx/Caddy.
 */
const API_PATHS = ["/chat", "/upload", "/threads", "/review", "/health"];

function buildProxy(apiKey) {
  const proxy = {};
  for (const path of API_PATHS) {
    proxy[path] = {
      target: "http://127.0.0.1:8000",
      configure: (instance) => {
        if (!apiKey) return;
        instance.on("proxyReq", (proxyReq) => {
          proxyReq.setHeader("Authorization", `Bearer ${apiKey}`);
        });
      },
    };
  }
  return proxy;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiKey = (env.AGENTFLOW_API_KEY || "").trim();
  const proxyTargets = buildProxy(apiKey);

  return {
    plugins: [react()],
    build: { sourcemap: true },
    server: {
      port: 5173,
      proxy: proxyTargets,
    },
    preview: {
      port: 4173,
      proxy: proxyTargets,
    },
    define: {
      // Surface VITE_API_BASE to the bundle so App.jsx can read it.
      "import.meta.env.VITE_API_BASE": JSON.stringify(
        env.VITE_API_BASE || ""
      ),
    },
  };
});
