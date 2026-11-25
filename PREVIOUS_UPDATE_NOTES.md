# V6 — taka branding and clean UI

- Removed both visual AI-readiness squares (header and Scanner heading). AI health polling and unavailable warning remain active.
- Replaced the B logo with a high-contrast vector Bangladeshi taka sign (৳), independent of installed browser fonts. Added matching browser favicon and included the public asset directory in the Docker frontend build.
- Updated the static design preview and removed obsolete readiness CSS. Model weights and the V5 65% threshold remain unchanged; the denomination update below adds OCR and backend validation.
- Rebuild frontend: `docker compose up -d --build --force-recreate web`. If running the complete extracted project, use `docker compose up -d --build --force-recreate`.

---

# V5 — full-note confidence fix

- Fixed V4's 90% full-image acceptance gate in both Python default and Docker Compose. New threshold: 65%. The classifier must still predict REAL or FAKE; not_banknotes remains rejected.
- Validation prediction CSV (3,811 rows): at 0.65, 703/708 fake-labelled and 957/966 real-labelled images accepted as either note class; 0/2,137 non-note images accepted. At 0.90 these counts were 678, 892 and 0. These are acceptance counts, not correct classification counts. Threshold selected on these same data; no independent calibration.
- Actual supplied checkpoint loaded via Keras 3.13.2 with PyTorch backend; on the visible banknote cropped from the supplied screenshot (not the original uploaded file), probabilities were REAL 0.122449, FAKE 0.844397, not_banknotes 0.033154. The new fallback gate accepts this prediction as FAKE. Original upload and Docker end-to-end inference not available here.
- Added full-image raw prediction logging and Java scan threshold/fakeProbability logging. No schema change, model weights unchanged.
- Rebuild with `docker compose up -d --build --force-recreate ai backend` and check `/health` reports `fullImageMinConfidence: 0.65`. Rescan; historical INVALID results are not changed.

---

# BankNoteGuard V4 — full-note inference fix

## Root cause and model provenance
The V3 AI service rejected every image with more than one surviving YOLO box before running EfficientNet. A box is not proof of a separate note. The new backup `banknote_final_backup(1).zip` contains the same best checkpoint as V3 (SHA-256 `3ab531b7c891cd8fa72a55adf1186dc9c299c93929f3e27aed63c4ebbb66f13d`). V4 copies the best checkpoint, mapping and summary directly from the new upload. The backup also includes a final checkpoint, not used. Original training reports are included in `training-report/`. The reported 99.2915% is validation accuracy from training, not V4 live performance.

## Server-side inference
Browser POST `/api/scans` -> Java Spring Boot `ScanService` -> Python FastAPI `/predict` -> YOLO candidate localization -> isolated TensorFlow EfficientNet-B0 `/classify` (localhost:8002) -> Java -> PostgreSQL and browser.

- Correct Ultralytics ndarray color order to BGR. Use class-agnostic NMS, plus overlap/containment consolidation to avoid duplicate box counting. Pad a selected crop by 4% of width/height on each side (clipped to image).
- One consolidated candidate: classify the crop with the existing three-class classifier; if classified as non-note, retry the entire image and accept only a REAL/FAKE prediction with confidence >= 0.65.
- Ambiguous multiple candidates or no candidate: classify the full upload; require >= 0.65 confidence for REAL/FAKE. A non-note or low-confidence result returns INVALID. The classifier uses 224x224 RGB float32 in 0–255; the model already contains rescaling.
- `VERIFY_MULTIPLE_NOTES=false` by default: the supplied single-class YOLO has not been validated to count complete notes. Raw box count no longer rejects a single-note scan. Therefore actual multi-note photos are not reliably rejected in this mode; scan one note per image. Optional `true` mode only rejects if two separated, plausible note-shaped crops both meet detector and classifier confidence thresholds. This heuristic is experimental, requires a validated full-note detector and real-photo evaluation; it can still miscount.
- `FULL_IMAGE_MIN_CONFIDENCE=0.65`, `MULTI_BOX_MIN_CONFIDENCE=0.65`, `MULTI_CLASS_MIN_CONFIDENCE=0.85` are configurable defaults; the 0.65 fallback threshold was selected using the saved validation predictions, not an independent full-note test set. A classifier may be overconfident on unknown objects; test false acceptance locally.
- Backend stores `inferencePath`, `detectorBoxCount`, `candidateCount` and logs them; nullable columns are added using existing Hibernate schema-update setting. Java validates AI response ID, source, status, denomination and confidence. Old records remain readable.
- Before V6, denomination stayed unknown when the current detector could not establish it. The backup does not contain 500/1000 denomination labels or new YOLO weights. Existing UI shows `500 Banknote` / `1000 Banknote` when a supported denomination-aware detector provides it. No fabricated denomination.
- V3 history thumbnails and Clear History retained. Clearing history removes DB rows but not original uploads or audit files. This local app has no user authentication; do not expose publicly without access controls.

## Run locally
Stop the old compose project in its own directory (`docker compose down`, never `down -v`). Extract V4, open inner `BankNoteGuard` directory, and run:

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose logs -f ai backend
```

Open http://localhost:5174 after http://localhost:8001/health shows `ready:true`. Hard refresh the browser. New health fields include `inferencePolicy`, `fullImageMinConfidence`, `verifyMultipleNotes`.

## Local acceptance checks
Use held-out photos of full real/fake 500 and 1000 notes, partial notes, non-notes, foreign currency, varied lighting, and multi-note photos. Inspect response `inferencePath` and box counts in backend logs; verify REAL/FAKE/INVALID and audit confidence. Test history image, refresh and clear history. Actual full-note accuracy and false acceptance must be measured locally. The supplied backup contains no original photos. Docker/TensorFlow/Ultralytics/Maven runtime not available in editing environment, so model loading, live inference and Java/frontend build were not run here. 20 mocked Python regression and API tests passed; ZIP/metadata/compose validation also passed. These checks are separate from live inference.

## V6 denomination and UI update
- Removed both decorative black readiness squares (header and scanner); textual readiness errors remain. Replaced B logo with taka symbol SVG and favicon.
- Added server-side Tesseract English/Bengali OCR on accepted notes, using a single bounded local OCR call on enlarged note regions. Exact supported numeric tokens and Bengali/English amount words only; conflicting/low-confidence evidence remains unknown. No effect on REAL/FAKE/INVALID, thresholds, or existing model weights. Future denomination-aware detector retains priority. OCR is optional at inference time: missing binary/timeouts do not break scans. Docker installs both language packs.
- Java validates and persists denomination; rejected responses must not contain an amount. Existing history is not retroactively updated.
- Actual screenshot crop returned denomination 500 with local OCR. 1000 amount parsing and API pipeline covered with regression tests; no independent 1000-banknote image set or full Docker runtime verification. OCR estimates are not guarantees, and reading a printed denomination does not authenticate a note.
