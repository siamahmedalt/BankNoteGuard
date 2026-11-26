# BankNoteGuard V16 — TensorFlow argmax

**Current instructions:** [V16_ARGMAX_TENSORFLOW.md](V16_ARGMAX_TENSORFLOW.md). Direct TensorFlow inference and highest-score classification are the defaults. No minimum classification confidence and no UNCERTAIN output for new scans. Older sections below are historical; their threshold and consensus rejection instructions are superseded.

# BankNoteGuard V15 — reference Python app integration

Read [V15_REFERENCE_INTEGRATION.md](V15_REFERENCE_INTEGRATION.md) for direct mode, comparative results and configuration. Consensus remains the default because the reference mode increased false FAKE results in the supplied samples. V14 confidence display is retained. Older notes below are historical.

# BankNoteGuard V14 — scan confidence display

Read [V14_CONFIDENCE_DISPLAY.md](V14_CONFIDENCE_DISPLAY.md) for the current output format and deployment instructions. The V13 model and inference policy are unchanged.

# BankNoteGuard V13 — training-aligned inference (based on supplied V10)

Read [V13_TRAINING_ALIGNMENT.md](V13_TRAINING_ALIGNMENT.md) first. This build uses the supplied training notebook's TensorFlow bilinear preprocessing and preserves the V10 consensus rule. Model weights and deployment ports are unchanged. Older version notes and diagnostics are historical.

# V8 classification review update

See [V8_FIX_NOTES.md](V8_FIX_NOTES.md) for current inference policy, diagnostic commands, and limitations. Older instructions below may describe earlier versions.

# BankNoteGuard V7 — integrated website rebuild

**Read [UPDATE_NOTES.md](UPDATE_NOTES.md) for the current detection policy, limitations, and regression checks.** The deployment instructions below remain applicable.

This package is based on BankNoteGuard V6, the latest replacement app.py, the backend fix bundle, and banknote_final_backup(2).zip and retains the minimal white square-corner interface, Scanner, History, React frontend, Spring Boot backend, PostgreSQL, YOLO checkpoint and EfficientNet-B0 model.

## Fixed

- Frontend Docker host port is **5174** (not 5173); your existing website on 5173 is unaffected.
- Vite development port is also **5174** with `strictPort`.
- Java Spring CORS allows `http://localhost:5174` and `http://127.0.0.1:5174`; the old whitelist caused possible `403 Invalid CORS request` on image POST.
- Python FastAPI is responsive while models load. `/health` reports `loading`, `ready`, `error`, and model-file presence. `/health/live` confirms server availability.
- React refreshes AI readiness every four seconds and displays actual error messages. Analyze is clickable when an image has been selected, even if the last AI health check was stale; a not-ready server replies with a descriptive 503 error.
- Java checks `/health` using a short timeout and allows longer image classification time. The message about manually starting Python on port 8000 was removed for Docker deployment.
- Nginx allows 300-second proxy response time for first-time CPU inference and accepts uploads up to 11 MB.
- YOLO confidence starts at 0.35 (configurable in `compose.yaml`) to help test low-confidence detections. Lower thresholds can increase false positives, so evaluate with real photos.
- AI Dockerfile installs CPU-only PyTorch to avoid unnecessary CUDA dependencies for standard Windows Docker Desktop / CPU deployment.
- The best three-class EfficientNet checkpoint is copied from the latest supplied backup. The legacy YOLO checkpoint is retained.

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
- A `REJECTED` scan is different from a 403: the AI pipeline could not verify the image as a banknote under the current inference policy. The uploaded model was trained with a single `banknote` class, not 500-vs-1000 labels. Tune `YOLO_CONFIDENCE` only after evaluating detections on real banknote photos.
- The classifier runs on the original upload for all normal scans. YOLO localizes notes but does not veto or override classification based on crop predictions. REAL and FAKE require confidence >= 0.70; non-banknotes and uncertain predictions are rejected. This threshold is not independently calibrated.

## Model limitations

`ai-service/models/banknotes_yolo.pt` recognizes a **generic banknote** class. It cannot certify that an image is a ৳500 or ৳1000 note or rule out all unsupported denominations. The new `.keras` model predicts REAL, FAKE or REJECTED for the full upload. Predictions are ML assessments, not conclusive currency authentication. V6 adds local Tesseract OCR (English/Bengali) to read printed 500/1000 amounts from accepted notes when YOLO supplies no denomination. This is a best-effort denomination estimate, not a verified denomination classifier. Unclear or conflicting OCR evidence remains null internally. The website displays Banknote for all accepted results; non-banknotes are not sent to OCR. For robust denomination recognition, train and independently validate a two-class YOLO model with labelled boxes for `500_taka` and `1000_taka`. Existing history is not retroactively changed. Re-scan notes with the new version.

## Data and history

Java stores results in PostgreSQL `scans` records. Python appends prediction metadata to a separate `prediction_results.pkl` inside the persistent `ai_data` Docker volume; it never overwrites `best_efficientnetb0_3class.keras`.

## Verification

46 Python regression/API tests passed with mocked model outputs. The actual best checkpoint also loaded and ran two screenshot-derived images with Keras 3.13.2 on the PyTorch backend. Full Docker image builds, TensorFlow worker inference, Java and React production builds remain unverified in this environment.


## V9 preprocessing correction
See `V13_TRAINING_ALIGNMENT.md` for current test evidence and limitations. `diagnostics/` contains historical CSVs and the V13 comparison. `CLASSIFIER_RESIZE_METHOD` must be `BILINEAR` for this checkpoint.
