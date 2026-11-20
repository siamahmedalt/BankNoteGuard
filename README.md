# BankNoteGuard — repaired Docker edition (localhost:5174)

This package is based on your uploaded `BankNoteGuard_Minimal_White_Updated(1).zip` and retains the minimal white square-corner interface, Scanner, History, React frontend, Spring Boot backend, PostgreSQL, YOLO checkpoint and EfficientNet-B0 model.

## Fixed

- Frontend Docker host port is **5174** 
- Vite development port is also **5174** with `strictPort`.
- Java Spring CORS allows `http://localhost:5174` and `http://127.0.0.1:5174`; the old whitelist caused possible `403 Invalid CORS request` on image POST.
- Python FastAPI is responsive while models load. `/health` reports `loading`, `ready`, `error`, and model-file presence. `/health/live` confirms server availability.
- React refreshes AI readiness every four seconds and displays actual error messages. Analyze is clickable when an image has been selected, even if the last AI health check was stale; a not-ready server replies with a descriptive 503 error.
- Java checks `/health` using a short timeout and allows longer image classification time. The message about manually starting Python on port 8000 was removed for Docker deployment.
- Nginx allows 300-second proxy response time for first-time CPU inference and accepts uploads up to 11 MB.
- YOLO confidence starts at 0.35 (configurable in `compose.yaml`) to help test low-confidence detections. Lower thresholds can increase false positives, so evaluate with real photos.
- AI Dockerfile installs CPU-only PyTorch to avoid unnecessary CUDA dependencies for standard Windows Docker Desktop / CPU deployment.
- Existing trained model weights stay unchanged.

## Requirements — Docker deployment

- Windows 10/11 with Docker Desktop **running**, Docker Compose v2 and sufficient disk/memory space.
- VS Code is optional. Chrome/Edge are recommended for camera permission.
- Internet during Docker build, because dependencies and Docker base images are downloaded.
- You do **not** need to install Java, Maven, Python, Node.js, or PostgreSQL separately when using Docker.

## Install / update on Windows

1. **Back up** your old `BankNoteGuard` folder and any important database data.
2. Open a PowerShell terminal inside your old *BankNoteGuard* project and run `docker compose down` (**not** `docker compose down -v`). This stops only the Compose project selected by that folder/name; do not stop the unrelated website on 5173.
3. Extract this ZIP and open its inner `BankNoteGuard` folder in VS Code. Confirm `compose.yaml` is in the current directory.
4. Run:

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps -a
```

5. Open **http://localhost:5174**, then hard refresh with `Ctrl+Shift+R` if an earlier frontend was cached.

For rebuild after a change:

```powershell
docker compose up -d --build ai backend web
```

For troubleshooting only, a full uncached rebuild is possible:

```powershell
docker compose build --no-cache ai backend web
docker compose up -d
```

**Never run `docker compose down -v` to update this project**; it deletes the PostgreSQL and AI data volumes.

## Services and ports

| Component | Windows host | Docker internal hostname |
|---|---|---|
| React website/Nginx | http://localhost:5174 | web:80 |
| Java Spring Boot | http://localhost:8081 | backend:8081 |
| Python FastAPI | http://localhost:8001 | ai:8000 |
| PostgreSQL | localhost:5433 | postgres:5432 |

IMPORTANT: do not change Java's `AI_SERVICE_URL=http://ai:8000` to port 8001. Port 8001 is exclusively the host port for Windows.

### Health and logs

```powershell
docker compose ps -a
curl.exe http://localhost:8001/health/live
curl.exe http://localhost:8001/health
curl.exe http://localhost:8081/api/health
curl.exe http://localhost:8081/api/system
docker compose logs --tail=150 ai
docker compose logs --tail=150 backend
```

While loading, Python reports `loading:true,ready:false`. Once models are loaded it reports `ready:true`. If `ready:false` and `loading:false`, inspect the `error` field and AI container logs. Startup depends on local machine performance and Docker memory.

To watch a scan progress:

```powershell
docker compose logs -f ai backend
```

You can also run the included PowerShell diagnostics from the project folder:

```powershell
.\DIAGNOSE.ps1
```

### If POST returns 403 again

- Verify the browser URL is **http://localhost:5174**, and rebuild both `backend` and `web`.
- Check browser Developer Tools → Network → `POST /api/scans` → Response; the exact reason is shown there.
- `CorsConfig.java` now includes localhost and 127.0.0.1 on port 5174. `localhost:5173` belongs to your other site and is intentionally **not allowed** to call this project's API cross-origin.
- Check Spring logs: `docker compose logs --tail=150 backend`.

### If Analyze returns 503 or an image is rejected

- A 503 while loading means the AI models aren't ready yet; inspect `http://localhost:8001/health`.
- A 503 with `error` in health means model initialization failed; review `docker compose logs ai`.
- A `REJECTED` scan is different from a 403: the YOLO model ran but did not detect precisely one supported banknote confidently enough. The uploaded model was trained with a single `banknote` class, not 500-vs-1000 labels. Tune `YOLO_CONFIDENCE` only after evaluating detections on real banknote photos.
- The real/fake classifier still runs only after YOLO has detected a banknote. It uses the supplied `.pkl` bundle and its existing 0.8044179 decision threshold.

## Model limitations

`ai-service/models/banknotes_yolo.pt` recognizes a **generic banknote** class. It cannot certify that an image is a ৳500 or ৳1000 note or rule out all unsupported denominations. Your `.pkl` model predicts REAL or FAKE for the cropped detected note. Predictions are ML assessments, not conclusive currency authentication. To display 500/1000 as a verified denomination, train a two-class YOLO model with labelled boxes for `500_taka` and `1000_taka`.

## Data and history

Java stores results in PostgreSQL `scans` records. Python appends prediction metadata to a separate `prediction_results.pkl` inside the persistent `ai_data` Docker volume; it never overwrites `best_efficientnetb0.pkl`.

## Verification

Source-level checks and local Python API/unit tests have been run. Full Docker image builds, Java/React production builds and real model inference must be confirmed using your Windows Docker Desktop system; those are not claimed as verified by this package.
