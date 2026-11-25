# V9 experimental preprocessing correction

- Default resize changed from LANCZOS to NEAREST in the API and classifier worker. The same `CLASSIFIER_RESIZE_METHOD` environment variable controls both. Set it to LANCZOS to revert interpolation; V9 also changes the decision policy.
- A confident primary REAL is kept. FAKE requires a confident primary FAKE and corroboration; alternate resize views cannot overwrite a weak or different primary label. Uncertain predictions remain review results. Threshold remains 0.70.
- Same checkpoint and class mapping; no retraining, reference-image matching or label overrides.
- On 20 user-labelled REAL images, raw checkpoint predictions: LANCZOS 4 REAL, 15 FAKE, 1 non-note; NEAREST 16 REAL, 4 FAKE, 0 non-note. Under V9 no-YOLO decision simulation using actual checkpoint outputs: 9 REAL, 3 FAKE, 8 review. One prior screenshot-derived user-labelled fake was FAKE 75.65%. This is not an independent accuracy or calibration study; labels are supplied by user, sample selected for troubleshooting, and fake/non-note coverage inadequate. Training-time interpolation not documented in backup. Both checkpoints had identical raw label counts under each tested common interpolation.
- 55 automated tests passed in local Python; these are primarily mocked tests. Actual inference used Keras with PyTorch backend, not deployed TensorFlow. YOLO and Docker end-to-end were not tested here.
- Permanent next step: confirm training loader and preprocessing and validate on an independent set of genuine and counterfeit physical notes, plus non-notes. Fine-tune on representative data with grouped splits and consistent preprocessing.

Run `docker compose up --build -d --force-recreate ai` from project root after stopping prior project. Open http://localhost:8001/health and check classifierResizeMethod is NEAREST.
