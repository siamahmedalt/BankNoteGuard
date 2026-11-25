# BankNoteGuard V7 — integrated rebuild

## Source and model
- Base: BankNoteGuard V6, latest corrected app.py, and backend replacement patch.
- Model and evaluation files from `banknote_final_backup(2).zip`. The best checkpoint SHA-256 is `3ab531b7c891cd8fa72a55adf1186dc9c299c93929f3e27aed63c4ebbb66f13d`, identical to backup (1). Only the best checkpoint is deployed, not the alternate final checkpoint. No retraining or accuracy improvement is claimed.
- EfficientNetB0, RGB 224x224, float32 pixel values 0–255 (rescaling inside model); real_notes=0, fake_notes=1, not_banknotes=2. Saved validation accuracy 99.29% (3,811 images), not a website test.

## Current inference policy
- YOLO confidence 0.35; consolidate duplicate/contained boxes. For normal scans use the full original image for authenticity, irrespective of the number of detection boxes. No conflicting crop prediction can veto it.
- REAL or FAKE requires confidence >= 0.70. Non-note and low-confidence results become INVALID. Optional multiple-note verification remains off; its thresholds are 0.90 detector and 0.85 classifier. It is not a validated counting method.
- OCR is optional to estimate denomination internally; OCR failure does not erase an accepted result. All accepted types display `Banknote`, rejected types display `Invalid`. Previous records still retain their original predictions.
- Previous UI fixes retained: taka logo, favicon, no decorative black boxes, scan history thumbnails, Clear History. Clear History removes database records, not AI audit files.
- Previous Java null-safety source fixes and Spring Boot 3.5.16 integrated. 3.5.x OSS support notice may persist; no major migration.

## Verification
- `cd ai-service; python -m pytest -q`: 46 passed (mocked pipeline/API, OCR parser and checkpoint metadata tests).
- Actual best checkpoint, Keras 3.13.2 / PyTorch backend, screenshot-derived crops resized to 224x224: 500 note from earlier fake-labelled screenshot predicted FAKE 78.98%; 1000 note portion from earlier real-labelled screenshot predicted REAL 98.10%. These are two examples, not independent accuracy estimates or original image files. They do not prove authenticity.
- Saved validation predictions at 0.70: 940/966 real-labelled images correctly accepted as REAL, 10 accepted as FAKE, 16 rejected; 691/708 fake-labelled images correctly accepted as FAKE, 8 accepted as REAL, 9 rejected; 0/2137 non-note-labelled images accepted. This is retrospective gating on saved predictions, not a new validation run.
- YAML, XML, Python syntax and ZIP checks performed. Docker, Maven and frontend production build not executed; TensorFlow deployment runtime not available. Use Docker Desktop to verify locally.

## Run
From the old project root: `docker compose down` (without `-v`). Extract V7 and open its inner BankNoteGuard folder. Run `docker compose up --build -d --force-recreate`, then `docker compose logs -f ai backend`. Wait for readiness at http://localhost:8001/health and open http://localhost:5174. Rescan previously rejected images. Keep volume backups.

See PREVIOUS_UPDATE_NOTES.md only for historical V3–V6 behavior, not current configuration.
