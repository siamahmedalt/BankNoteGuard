# V15 — reference Python app integration

Based on V14, with the inference operations from the supplied `Banknote_Detection-main/app-LAPTOP-ADMVLS6C.py` adapted to the existing FastAPI worker. The Flask HTML/server is retained for reference only under `reference-app/`; do not launch it as part of Docker Compose. No frontend, database schema or model weight replacement is required.

## Model and preprocessing

Both three-class checkpoints have exactly the same SHA-256:
`3ab531b7c891cd8fa72a55adf1186dc9c299c93929f3e27aed63c4ebbb66f13d`.
The mapping is real_notes=0, fake_notes=1, not_banknotes=2. The reference app decodes original bytes with `tf.io.decode_image`, performs a floating-point TensorFlow bilinear resize to 224x224, keeps 0–255 RGB float32, then runs a single softmax prediction and argmax. The V14 worker already used the same resize but transported EXIF-corrected RGB PNG views. In direct mode, JPEG/PNG original bytes are sent verbatim to the isolated worker, matching the reference app (including ignoring EXIF orientation for classification). WebP is decoded using Pillow, EXIF corrected and sent as RGB PNG because TensorFlow decode_image does not support WebP. The browser preview and YOLO localization can be EXIF-corrected even when direct classification uses original bytes. Direct mode still applies BankNoteGuard's existing 70% minimum confidence rule; the reference app itself does NOT apply a threshold. Raw argmax and all probabilities are logged and available via the diagnostic script. The other binary checkpoint and its binary threshold are not used by the requested reference script and are deliberately not deployed.

## Choose inference mode

`compose.yaml` -> ai -> environment:

* `CLASSIFIER_INFERENCE_MODE: "consensus"` (default): preserve V14 rotation/crop aggregation, 8 views, threshold 0.70, at least 3 votes per family.
* `CLASSIFIER_INFERENCE_MODE: "direct"`: single-view original-byte prediction with threshold 0.70; no crop/rotation consensus. YOLO remains for localization and optional multi-note verification, not authenticity. Direct mode can be faster but is more sensitive to rotation/cropping and may make confident mistakes.

Change the value, then `docker compose up -d --build --force-recreate ai`. Check `http://localhost:8001/health` for `classifierInferenceMode` and `classifierModelSha256`. Launch the entire project with `docker compose up --build -d --force-recreate`; website: http://localhost:5174. Do not use `down -v` unless you intend to delete the database and history volumes.

## Evidence and limitations

`diagnostics/v15_reference_comparison.csv`: actual shared Keras checkpoint, six previously supplied images labelled REAL by the user (not independently authenticated). Executed with Keras PyTorch backend, Pillow decoding and PyTorch bilinear interpolation (align_corners=False, antialias=False), approximating TensorFlow preprocessing; NOT a production TensorFlow or Docker run. Direct guarded: 1 REAL, 3 FAKE, 2 UNCERTAIN. V14 consensus: 1 REAL, 1 FAKE, 4 UNCERTAIN. Raw reference argmax: 3 REAL, 3 FAKE. Therefore default remains consensus rather than introducing extra false FAKE outputs. These are not overall accuracy estimates; no independent FAKE/non-note test set was provided for this comparison. No retraining or guaranteed authentication improvement. A physical authenticity check or verified retraining dataset is still necessary.

To compare exact deployed TensorFlow predictions, copy a local image into the running AI container and run:

```
docker compose cp "C:\path\to\note.jpg" ai:/tmp/note.jpg
docker compose exec ai python diagnose_image.py /tmp/note.jpg
```

This prints configured policy, direct raw argmax and guarded result with model checksum. No history is added. `check_stability.py` compares transformed images using the configured policy, without original JPEG byte parity for generated variants.

Tests: 86 Python tests passed, 3 TensorFlow numerical parity tests skipped (TensorFlow unavailable); 6 frontend Node tests passed. No Maven, Docker, or React production build verification in this environment. Original model and V14 UI confidence fields retained.
