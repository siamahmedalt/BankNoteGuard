# V13 — Training-aligned inference (supplied V10 base)

## Sources and decisions
- Website source: `BankNoteGuard_v10(1).zip` (frontend, Java backend, YOLO and database configuration preserved).
- Training source: `training/build-efficientnetb0-final.ipynb` (original uploaded notebook, retained unchanged).
- Model and reports: `banknote_final_backup(3).zip`. All CSV/PNG reports and model summary/class mapping are retained. The best checkpoint is the deployed file. Its SHA-256 is unchanged from the supplied V10. Both uploaded Keras checkpoints contain identical values for all 314 weight tensors, as verified by loading them with Keras 3.13.2 / PyTorch CPU. The second checkpoint is not duplicated in the website ZIP.
- Training uses `tf.io.decode_image(..., channels=3, expand_animations=False)`, `tf.image.resize(..., [224,224])` (default bilinear, antialias=False) and float32, 0–255 input. EfficientNet contains internal preprocessing. No external division by 255 is added.
- V10 resized in the API with Pillow NEAREST and sent already resized 8-bit pixels. V13 transfers native RGB PNG losslessly and performs exactly one floating-point resize using TensorFlow in the isolated worker. Pillow handles original uploads, EXIF and WebP as before, so JPEG decode / EXIF behavior may differ from the notebook's direct file decoding. Rotation and mild crop remain inference-time transformations, not training augmentations.
- The training-derived `models/inference_contract.json` binds class order, shape, pixel range, interpolation and deployed checkpoint checksum. Startup fails on incompatible manifest/weights. If deliberately deploying newly validated weights, update the manifest's SHA-256 and other fields only after verifying the corresponding training contract.
- V10's eight-view symmetric consensus and 0.70 threshold are preserved. This is not a recalibrated decision rule. A higher threshold can reduce some false claims but cannot fix a classifier that mislabels a note.

## Reported validation versus real-world tests
The original notebook/backup report 99.29% validation accuracy, macro F1 98.97%, 13,206 train and 3,811 validation images. There is no independent test dataset or retraining in this release. No random rotation or crop augmentation is present in the original training code. The banknote group split and Caltech category split are retained as recorded; no independent leakage verification was possible without the full data.

`diagnostics/v13_preprocessing_comparison.csv` compares old NEAREST to training-style floating-point bilinear (PyTorch interpolate, align_corners=False, antialias=False) with the supplied actual checkpoint, Keras PyTorch CPU. This is not TensorFlow production execution or a full YOLO/website test. User labels, not independently authenticated notes, are the ground truth. On eight prior REAL-labelled uploads, old policy returned 3 REAL / 5 UNCERTAIN; new preprocessing returned 3 REAL / 4 UNCERTAIN / 1 FAKE. On the most recent six samples, the new policy returns 1 REAL / 4 UNCERTAIN / 1 FAKE. Thus the mismatch is fixed in source, but authenticity accuracy has NOT improved on these samples and one known false-FAKE regression exists. Do not use this build as a certified authentication system; it is a training-aligned experimental build. Never force these samples to REAL or add hash-based exceptions. Retraining with verified representative REAL and FAKE notes, rotation/crop/lighting augmentation, group separation, and independent evaluation is required to establish accuracy.

## Run on Windows
1. In old project directory: `docker compose down` (do not add `-v` if retaining local database/history).
2. Extract this ZIP, open the inner `BankNoteGuard` directory.
3. `docker compose up --build -d --force-recreate`
4. `docker compose logs -f ai backend`
5. `start http://localhost:5174`
6. Check `http://localhost:8001/health` for ready=true, classifierResizeMethod=BILINEAR, and correct classifierModelSha256.

For an original-image diagnostic with the production TensorFlow worker:
```
docker compose cp "C:\path\to\note.jpg" ai:/tmp/note.jpg
docker compose exec ai python diagnose_image.py /tmp/note.jpg
```
The existing diagnostic logs per-view probabilities and consensus. Use `check_stability.py` to compare rotations and trims. Native-resolution PNG transfer may increase latency for large uploads; the worker is localhost only.

## Tests
`cd ai-service && python -m pytest -q`: 70 passed, 1 skipped (TensorFlow-specific numerical parity test). TensorFlow, Docker and Maven are not available in the build environment; Java/frontend builds and full end-to-end inference are not verified here. The PyTorch-backend sample audit is separate from the mocked API tests. Run the included TensorFlow parity test in an environment with TensorFlow and pytest installed before deployment.
