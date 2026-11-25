# BankNoteGuard V8 — classification review and diagnostics

## Findings
The uploaded best EfficientNetB0 checkpoint has built-in 1/255 rescaling; external normalization is not required. Metadata labels are 0 REAL, 1 FAKE, 2 non-banknote. V7 rejected original-image REAL/FAKE scores below 0.70. Its classifier itself can misclassify images: recorded validation has 12/966 REAL images classified FAKE and 10/708 FAKE images classified REAL. These are original validation records, not V8 performance.

The current screenshot-derived 1000 BDT crop produces REAL 98.2772% using the same checkpoint in Keras/PyTorch; V7's original-image policy would also accept that input. This does not reproduce the user's local error. Original uploaded bytes and Docker TensorFlow output are needed to identify the cause of that scan. Inference from the screenshot is not proof of authenticity. Rotating that crop 90 degrees changed the prediction substantially, showing orientation sensitivity.

## Changes
- Keep the original-image confidence threshold at 0.70; do not force a FAKE prediction into REAL.
- Preserve confident original REAL predictions. For FAKE and uncertain predictions, check original Lanczos, bilinear, and bicubic resize or a sufficiently large YOLO crop. Two strong agreeing views without strong opposition are required. For clearly portrait inputs, two quarter-turn checks can veto contradictory decisions; square close-ups are not quarter-turned. Strong non-note consensus still yields INVALID. Conflicting/weak evidence yields UNCERTAIN (wire status REJECTED with review_uncertain path, preserving DB schema).
- Scores for accepted multi-view results are mean scores across strong agreeing views, not calibrated authenticity probabilities. These correlated views are not independent evidence. This provisional policy may increase uncertain outputs and is not validated on the original dataset; it cannot repair a model whose views all confidently misclassify.
- Worker reports all 3 probabilities in VIEW logs, loaded model SHA-256 and TensorFlow version in health. No change to weights, database, Java contracts, UI logo, history or clear-history controls. Added diagnostic script.

## Run on Windows
In the old project folder: `docker compose down` (do not add `-v`). Extract this ZIP; open its BankNoteGuard folder. Run:

```
docker compose up --build -d --force-recreate
docker compose logs -f ai
```

Open http://localhost:5174, refresh with Ctrl+Shift+R. Check http://localhost:8001/health for `inferencePolicy: v8-conservative-multiview-review` and the loaded model SHA-256. The expected SHA-256 is in `training-report/v8_screenshot_diagnostics.json`. Scan the exact original file again. For each `VIEW` entry, the log shows REAL, FAKE and not-banknote scores. The frontend displays UNCERTAIN separately from INVALID.

## Diagnose an exact original file
Copy the original dataset image, without screenshotting or resaving, into the running AI container (replace the example filename):

```
docker compose cp "C:\path\to\real_note.jpg" ai:/tmp/original.jpg
docker compose exec ai python diagnose_image.py /tmp/original.jpg
```

This uses the running TensorFlow worker with the original image, without YOLO crop, and does not add history. Compare its raw scores, EXIF orientation, model hash and threshold with the website's VIEW logs. If it confidently predicts FAKE for genuine notes, collect the original mislabeled samples and retrain/fine-tune using correct labels, capture conditions, orientations, full images and partial crops. Split by physical note identity to avoid leakage; validate on held-out physical notes. Do not use photo-only AI as definitive currency authentication.

## Testing
53 automated regression/API/OCR tests passed (mocked model outputs). The checkpoint was loaded via Keras/PyTorch and three screenshot-derived samples were evaluated with the new decision policy and without YOLO; details in `training-report/v8_screenshot_diagnostics.json`. TensorFlow/Docker, frontend and Java production build, and original dataset evaluation were not run here.
