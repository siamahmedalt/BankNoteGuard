# BankNoteGuard

**Bangladeshi Banknote Scanner — React + Java Spring Boot + YOLO11 + EfficientNet-B0 + PostgreSQL**

A prototype for scanning banknote images using file upload or camera capture. The minimal, square-corner interface shows **Scanner** and **History** side by side. A YOLO11 **object detector** selects a banknote region; the EfficientNet-B0 classifier predicts **REAL** or **FAKE**; the Java API stores scan records in PostgreSQL.

> **Research/demo only.** This machine-learning prediction is not an official authenticity check. The included YOLO weights detect a generic **banknote** class; they do **not** prove that the note is specifically ৳500 or ৳1000. Real/fake predictions are subject to training-data quality and errors.

## Project structure

```text
BankNoteGuard/
├── frontend/                    # React + Vite; Nginx for Docker locally
├── backend/                     # Java Spring Boot API
├── ai-service/                  # FastAPI + YOLO11 + isolated EfficientNet worker
│   ├── app.py
│   ├── classifier_worker.py
│   └── models/
│       ├── banknotes_yolo.pt
│       └── best_efficientnetb0.pkl
├── compose.yaml                 # LOCAL Docker Compose, not Render configuration
└── README.md
```

## Local Docker — Windows / VS Code

Install Docker Desktop, then open the **inner `BankNoteGuard` folder containing `compose.yaml`** in VS Code:

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps -a
```

| Local component | URL / port | Docker-to-Docker address |
|---|---|---|
| React website | http://localhost:5174 | `web:80` |
| Java Spring Boot | http://localhost:8081 | `backend:8081` |
| Python AI | http://localhost:8001 | `ai:8000` |
| PostgreSQL | `localhost:5433` | `postgres:5432` |

Check readiness:

```powershell
curl.exe http://localhost:8001/health
curl.exe http://localhost:8081/api/health
curl.exe http://localhost:8081/api/system
docker compose logs --tail=100 ai backend
```

`ai.ready=true` means the trained detector and classifier have loaded. `Up` alone does not prove they are ready. Use `docker compose down` to stop; **do not run `docker compose down -v`** unless intentionally deleting the named volumes and their data.

**Local API routing:** the React build uses relative `/api/...` paths by default. Local Nginx proxies requests to `http://backend:8081`. The Java-to-Python URL is `http://ai:8000` **inside Docker**; `localhost:8001` is only for Windows host access.

## GitHub — updating the existing repository

Repository: [siamahmedalt/BankNoteGuard](https://github.com/siamahmedalt/BankNoteGuard) (private or public according to the owner's GitHub settings).

**Recommended:** commit changes from your *existing Git clone*, not from a freshly extracted ZIP without `.git`. The original project has been accessed locally at `D:\400C\BankNoteGuard_V1\BankNoteGuard`.

```powershell
cd "D:\400C\BankNoteGuard_V1\BankNoteGuard"
git status -sb
git pull --ff-only origin main
```

Copy only reviewed files from the updated ZIP into that clone (preserving the same relative paths). Inspect the difference:

```powershell
git status --short
git diff --check
git diff -- README.md frontend/src/App.jsx frontend/.env.example
```

Stage **only** the intended files (not `compose.yaml` or secrets):

```powershell
git add README.md frontend/src/App.jsx frontend/.env.example
git diff --cached --stat
git diff --cached --check
git commit -m "Document Render deployment and configure frontend API URL"
git push origin main
```

Confirm push:

```powershell
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status -sb
```

The two `rev-parse` hashes should match. If there are additional actual source differences (`ai-service` etc.), review with `git diff` and stage them explicitly; **do not blindly `git add .`**. Never commit `.env`, credentials, private scan uploads or databases. The `.pt` and `.pkl` files are **trained model artifacts**, not new output files to generate during Render deployment.

## Render deployment — GitHub to public website

This is a **multi-service** deployment. Render does **not** run `compose.yaml` as a single Docker Compose project. The existing three hosted services and Postgres database are managed separately. These are example service URLs based on the existing deployment; verify actual addresses in your Render Dashboard.

```text
Browser
  └── https://banknoteguard-frontend.onrender.com
       React / Vite static website
         └── HTTPS API requests
              └── https://banknoteguard-backend.onrender.com
                   Java Spring Boot REST API
                     ├── Render PostgreSQL (private DB hostname)
                     └── HTTPS inference requests
                           └── https://banknoteguard-ai.onrender.com
                                Python FastAPI + YOLO11 + EfficientNet-B0
```

### A. React frontend — Static Site

Create/connect the existing GitHub repository on Render → **New → Static Site**:

| Setting | Value |
|---|---|
| Service | `banknoteguard-frontend` |
| Branch | `main` |
| Root Directory | `frontend` |
| Build Command | `npm install && npm run build` |
| Publish Directory | `dist` |
| Plan | Free static site |
| Environment: `VITE_API_BASE_URL` | `https://banknoteguard-backend.onrender.com` |

**Render UI note:** when Root Directory is `frontend`, the UI may display `frontend/` automatically in front of the command/directory fields; do not type it again.

`frontend/src/App.jsx` uses `VITE_API_BASE_URL` when set and relative `/api/...` locally when empty. `VITE_` variables are included **at build time**, so after adding/changing this value use **Save, rebuild and deploy**. Do not add confidential secrets to `VITE_` variables; they are exposed to browser code.

A static-site *SPA rewrite* from `/*` to `/index.html` may be useful if client-side routes are added. It is **not** a substitute for routing `/api` requests to a real backend.

### B. PostgreSQL — Render Postgres

Your Render database is named `banknoteguard-db`, database name `banknoteguard`, and is in **Oregon (US West)**. Use its current **Internal Hostname**, username and freshly rotated password from the Render Dashboard. Do **not** commit connection strings with credentials.

The Render Postgres **Internal Database URL** generally starts with `postgresql://...`; this Spring Boot app expects a **JDBC** URL in the form `jdbc:postgresql://HOST:5432/banknoteguard`.

### C. Java backend — Docker Web Service

| Setting | Value |
|---|---|
| Service | `banknoteguard-backend` |
| Root Directory | `backend` |
| Runtime | Docker |
| Dockerfile Path | `Dockerfile` (relative to root directory in Render UI) |
| Region | Oregon (US West), same as Postgres |
| Instance | Free (subject to memory limits) |

Set these backend **Environment Variables** using the Dashboard, **not** a committed file:

| Variable | Value / purpose |
|---|---|
| `PORT` | `8081` |
| `DATABASE_URL` | `jdbc:postgresql://YOUR_RENDER_INTERNAL_HOST:5432/banknoteguard` |
| `DATABASE_USER` | Current Render Postgres username |
| `DATABASE_PASSWORD` | Current Render Postgres password |
| `APP_CORS_ORIGINS` | `https://banknoteguard-frontend.onrender.com` (no trailing `/`) |
| `AI_SERVICE_URL` | `https://banknoteguard-ai.onrender.com` (no trailing `/health`) |

Spring Boot reads `DATABASE_URL`, `DATABASE_USER`, `DATABASE_PASSWORD`, `PORT` and `AI_SERVICE_URL` through `backend/src/main/resources/application.properties`; `CorsConfig.java` reads `APP_CORS_ORIGINS` via Spring's relaxed property binding. Click **Save, Deploy** after changes. Check Java logs for `HikariPool-1 - Start completed` and `Started BankNoteGuardApplication`.

### D. Python AI — Docker Web Service

| Setting | Value |
|---|---|
| Service | `banknoteguard-ai` |
| Root Directory | `ai-service` |
| Runtime | Docker |
| Dockerfile Path | `Dockerfile` |
| Region | Oregon (US West) |
| Instance | Free **only if runtime memory allows** |

Environment variables:

| Variable | Value |
|---|---|
| `PORT` | `8000` |
| `BANKNOTE_MODEL_PATH` | `/app/models/best_efficientnetb0.pkl` |
| `YOLO_WEIGHTS_PATH` | `/app/models/banknotes_yolo.pt` |
| `STORAGE_DIR` | `/app/storage` |

The Dockerfile's Uvicorn command binds `0.0.0.0:8000`. `classifier_worker.py` is required: the EfficientNet/TensorFlow classifier runs in a separate process to avoid a known TensorFlow/PyTorch native runtime conflict. Both processes still share the **same Render instance RAM**.

`PORT=8000` is the external HTTP listener for FastAPI. A classifier subprocess listening only on loopback does **not** require a separate Render service or exposed external port.

### E. After every GitHub push

When Auto-Deploy is enabled, Render may deploy linked services from the `main` branch automatically. Verify each deploy's commit SHA instead of assuming it succeeded. Otherwise:

1. `banknoteguard-ai` → **Manual Deploy → Deploy latest commit**, then check `/health`.
2. `banknoteguard-backend` → deploy, then check `/api/health`, `/api/system`, `/api/scans`.
3. `banknoteguard-frontend` → deploy, then check website Upload, Camera, Analyze and History.

Check actual URLs:

- **AI health:** https://banknoteguard-ai.onrender.com/health — must show `ready: true` for inference.
- **Java health:** https://banknoteguard-backend.onrender.com/api/health — `api: online`.
- **Java-to-AI status:** https://banknoteguard-backend.onrender.com/api/system — `ai.ready: true` for complete inference chain.
- **Scan history:** https://banknoteguard-backend.onrender.com/api/scans — expected to be an array.
- **Frontend:** https://banknoteguard-frontend.onrender.com — public UI.

If the browser shows `Java backend is not responding`, inspect its Network tab and `VITE_API_BASE_URL` build setting. If the API reports `Python AI is not reachable through Docker network` / `localhost:8000`, configure **the backend service being used by the frontend** with the correct `AI_SERVICE_URL`; an old Java error message might mention Docker even on Render.

### Free-hosting limitations and persistence

Render Free Web Services currently provide **512 MB RAM and 0.1 CPU**, can spin down after about **15 minutes** without traffic, and may be slow to wake up. **YOLO11 + PyTorch + TensorFlow EfficientNet-B0 is likely too heavy for this memory limit.** An HTTP service initially showing `online` or `loading` does not mean both models will finish loading; `502 Bad Gateway` after loading may indicate an OOM crash or another startup failure. Review Render → AI → **Events / Logs / Metrics**; check memory, exit status and traceback. Do not claim full AI inference is working until `ready:true` and a real scan succeed on Render.

Render Free Postgres **expires after 30 days** and is not a durable database for long-term use. Back up scan history before expiration. Local files inside free web services are **ephemeral**; `/app/storage/prediction_results.pkl` may disappear after service restart, spin-down or deploy. PostgreSQL scan history is the supported persistent record for the lifespan of the hosted database. The project has **not** been validated for financial decisions.

**Security:** rotate any Render Postgres password previously exposed in screenshots or committed files; keep credentials only in Render's private Environment settings. The default `compose.yaml` password is a local development example and must **never** be reused for a hosted database.

## Common troubleshooting

| Symptom | Check / fix |
|---|---|
| `no configuration file provided` | In VS Code, `cd` to the inner folder containing `compose.yaml`. |
| `additional properties ... not allowed` | Validate `compose.yaml` with `docker compose config --quiet`; `services:` must be the top-level key. |
| Local UI says AI offline | `curl.exe http://localhost:8001/health` and `curl.exe http://localhost:8081/api/system`. |
| Render UI says backend offline | Check Render frontend `VITE_API_BASE_URL` and browser Network errors; CORS must allow exact frontend origin. |
| `403 Invalid CORS request` | Confirm `APP_CORS_ORIGINS` exactly matches the deployed `https://...` frontend origin. |
| AI service `502` | Inspect Render AI logs/metrics for OOM, process exit, port binding and model initialization errors. |
| `LLVM ERROR` / `bitcode-mdindex-threshold` | Confirm `classifier_worker.py` and corresponding isolated-worker `app.py` are deployed from the same Git commit. |
| Image `REJECTED` | YOLO did not find exactly one supported note; this is different from a network error. |
| Backend `/` says `Unexpected server error` | Use `/api/health` or `/api/system` for API health; `/` is not an API route in this project. |

## Official references

- [Render: Web services](https://render.com/docs/web-services)
- [Render: Docker](https://render.com/docs/docker)
- [Render: Environment variables](https://render.com/docs/configure-environment-variables)
- [Render: Free-plan limits](https://render.com/docs/free)
- [Render: Troubleshooting deploys](https://render.com/docs/troubleshooting-deploys)
