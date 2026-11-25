# Local Docker AI crash fix

This update keeps the React frontend at localhost:5174, Java at localhost:8081, PostgreSQL at localhost:5433 and Python AI at localhost:8001.

## What changed

YOLO / PyTorch runs in the main FastAPI process. EfficientNet / TensorFlow loads in a separate localhost-only child process (port 8002 inside the AI container, **not exposed to Windows**). This isolates native runtimes to mitigate `LLVM ERROR: Option bitcode-mdindex-threshold registered more than once`.

No model weights, Java endpoints or saved database volume are modified. The classifier's original RGB 224×224 preprocessing and decision threshold are preserved.

## Apply on Windows

1. Back up your current `BankNoteGuard` folder.
2. Extract this ZIP and use its `BankNoteGuard` folder, preserving the existing database volume.
3. From the folder containing `compose.yaml` run:

```powershell
docker compose up -d --build ai
```

You may rebuild all services if you replaced the whole project (`docker compose up -d --build`), but do not use `docker compose down -v`.

## Verify

```powershell
docker compose logs -f ai
```

Wait for `ALL MODELS READY` and `EfficientNet isolated worker READY`, then in another terminal:

```powershell
curl.exe http://localhost:8001/health
curl.exe http://localhost:8081/api/system
```

Both should have `ready:true`. Open http://localhost:5174 and analyze a note.

To confirm no ongoing crashes:

```powershell
docker inspect banknoteguard-ai-1 --format '{{.RestartCount}}'
```

Run it again after a few successful scans; the count should not rise. If it does, capture the latest `docker compose logs --tail=150 ai` and check Docker Desktop memory settings.

## Limitations

This is a targeted process-isolation fix, not a promise that every crash is solved. It may consume more memory and does not fix Render free-tier RAM limits. The YOLO detector supplied is single-class and cannot distinguish denominations.
