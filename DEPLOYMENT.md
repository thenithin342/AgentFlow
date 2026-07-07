# Deployment Guide for AgentFlow

Deploying AgentFlow involves two main parts: the **FastAPI Backend** and the **React Frontend**. Because the application uses SQLite for short-term memory (Checkpointer) and local files (FAISS/Pickle) for Long-Term Memory (LTM), the easiest and most robust way to deploy is using **Docker** and **Docker Compose** on a single Virtual Machine (VM) like an AWS EC2 instance, DigitalOcean Droplet, or Google Cloud Compute instance.

Alternatively, you can deploy the frontend to a static host (Vercel/Netlify) and the backend to a PaaS (Render/Railway), but you will need to handle persistent storage for the backend databases.

---

## Option 1: Docker Compose on a Virtual Machine (Recommended)

This approach bundles both the frontend and backend into containers and runs them on a single server. It natively handles the local file storage needed for `agentflow.db` and the `ltm_indexes` folder.

### 1. Prerequisites
- A Linux VM (e.g., Ubuntu on AWS EC2 or DigitalOcean).
- Docker and Docker Compose installed on the VM.
- A registered domain name (optional, but recommended for production HTTPS).

### 2. Create Docker Configuration Files
Create the following files in the root of your project:

**`backend.Dockerfile`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Expose FastAPI port
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`frontend.Dockerfile`**
```dockerfile
# Build stage
FROM node:18-alpine as build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
# Adjust the backend URL for production
ENV VITE_API_URL=/api 
RUN npm run build

# Serve stage (Nginx)
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Setup basic nginx configuration to route /api to the backend
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

**`nginx.conf`**
```nginx
server {
    listen 80;
    
    location / {
        root   /usr/share/nginx/html;
        index  index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s; # Increase timeout for long LLM responses
    }
}
```

**`docker-compose.yml`**
```yaml
version: '3.8'

services:
  backend:
    build: 
      context: .
      dockerfile: backend.Dockerfile
    volumes:
      - ./agentflow.db:/app/agentflow.db
      - ./ltm_indexes:/app/ltm_indexes
    env_file:
      - .env
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: frontend.Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

### 3. Deploy to the Server
1. Clone your repository onto the VM.
2. Create your `.env` file on the server with your production keys (e.g., `OPENAI_API_KEY`).
3. Run the following command in the root directory:
   ```bash
   docker-compose up -d --build
   ```
4. The app is now running on your server's public IP address.

---

## Option 2: Railway (single-service backend deploy)

The fastest deploy path that doesn't require a VM. Railway runs a single
container built from the repo's `Dockerfile`, attaches a managed Postgres
plugin for the checkpointer, and gives you a public HTTPS URL on
push-to-main.

### What you get
- One backend service (the FastAPI image)
- One Postgres database (auto-injected as `DATABASE_URL` / `POSTGRES_CONN_STRING`)
- One static-frontend service (deployed separately to Railway static hosting, Vercel, or Netlify — frontend is NOT in this image)
- HTTPS + a stable URL on every successful deploy
- Health-check-gated deploys (failed health probes roll back automatically)

### Step 1 — Create the Railway project
1. Sign in at https://railway.app and click **New Project → Deploy from GitHub repo**.
2. Select the `agentflow` repository. Railway will scan for `railway.toml`
   and pick up our Dockerfile build config.

### Step 2 — Add Postgres
1. In the project canvas click **+ New → Database → Postgres**.
2. Once it's provisioned, click the backend service, then **Variables →
   New Variable → Add Reference** and select `DATABASE_URL` from the
   Postgres service. Rename the reference to `POSTGRES_CONN_STRING`
   so it matches what `backend/settings.py` reads.

### Step 3 — Set the required secrets
On the backend service's **Variables** tab, add:
| Name | Value | Notes |
| --- | --- | --- |
| `ENVIRONMENT` | `production` | Triggers fail-fast validation of secrets |
| `JWT_SECRET` | output of `openssl rand -hex 32` | 32+ random bytes |
| `GROQ_API_KEY` | your Groq key | Required at first request |
| `TAVILY_API_KEY` | your Tavily key | Required when search agent runs |
| `CORS_ORIGINS` | `https://your-frontend.up.railway.app` | Comma-separated; add both staging + prod URLs |
| `ADMIN_PASSWORD` | long random string | Change from default; bootstrap admin created on first boot |

Optional: `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true` for trace
capture; `GROQ_API_KEY_2`, `GROQ_API_KEY_3` for key rotation.

### Step 4 — Add a persistent volume
The default container filesystem is ephemeral. For durable FAISS /
LTM indexes:
1. Backend service → **Settings → Volumes → New Volume**.
2. Mount path: `/app/data` (covers `faiss_indexes/` and `ltm_indexes/`
   via the defaults in `backend/settings.py`).

### Step 5 — Deploy
Push to `main`. Railway builds the Docker image, runs the healthcheck
against `/healthz`, and on success returns the public URL. Tail logs from
the **Deployments** tab to confirm `uvicorn` is up.

### Step 6 — Point the frontend at the backend
In the frontend service's environment, set:
```
VITE_API_URL=https://<your-backend>.up.railway.app
```
Rebuild the frontend. The Vite dev proxy pattern (never bundle keys
into client JS) still applies — see Option 3 for the full frontend
walkthrough.

### Notes
- We use `DOCKERFILE` builder (not Nixpacks) so the deployed image
  matches `docker-compose.yml` byte-for-byte. To force Nixpacks instead
  set `BUILDER=NIXPACKS` and remove the `[build]` block from
  `railway.toml`.
- `railway.toml` declares `healthcheckPath = "/healthz"`. This is the
  cheap liveness probe (no DB, no graph). `/readyz` exists for k8s-
  style orchestrators but Railway only honors one healthcheck path.
- Cold start is ~60–90s the first time (sentence-transformer model
  download). Warm restarts are < 5s.

---

## Option 3: PaaS (Vercel + Render/Railway)

If you don't want to manage a VM, you can split the app across managed platforms.

### 1. Frontend (Vercel or Netlify)
1. Push your code to GitHub.
2. Go to Vercel/Netlify, import your repository, and select the `frontend` folder as the Root Directory.
3. Add an Environment Variable: `VITE_API_URL = https://your-backend-url.onrender.com`
4. Deploy.

### 2. Backend (Render or Railway)
Because AgentFlow writes to local files (`agentflow.db` and `ltm_indexes`), **you must mount a persistent disk**. Ephemeral disks on PaaS providers wipe out files on every deployment, which would delete all agent memory.

**On Render:**
1. Create a new **Web Service** connected to your repo.
2. Set the Build Command: `pip install -r requirements.txt`
3. Set the Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
4. **Important**: Go to the "Disks" section and add a disk mounted at `/opt/render/project/src/data`.
5. You will need to update your backend code to save the `.db` and `ltm_indexes` into that specific folder if running in production.
6. Add your Environment Variables (like `OPENAI_API_KEY` and your basic auth `API_KEY`).

### Security Reminders for Deployment:
- **Enable Auth**: Make sure your `API_KEY` environment variable is set to a strong, secure value in your production `.env` so the `backend/security.py` middleware protects your endpoints.
- **HTTPS**: If using Option 1, use an Nginx reverse proxy with `certbot` to provision free SSL certificates via Let's Encrypt so traffic is encrypted.
- **Timeouts**: LLMs can take time to respond, especially the blog agent. Ensure your reverse proxy (Nginx or PaaS) has a high timeout limit (e.g., 5 minutes).
