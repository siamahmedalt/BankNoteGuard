# V16 — TensorFlow argmax classification

The deployed classifier is the existing three-class EfficientNetB0 checkpoint, with the original training preprocessing: RGB, float32, 0–255, TensorFlow bilinear resize to 224×224. The Docker requirements already pin tensorflow-cpu==2.20.0 and keras==3.13.2; this release explicitly sets KERAS_BACKEND=tensorflow and checks the backend at startup. The VS Code TensorFlow extension is not the runtime.

## Decision

For every successful classification of a supported image, choose the largest of real_notes, fake_notes, not_banknotes (in that class-ID order for exact ties). Return REAL, FAKE, or REJECTED (displayed as INVALID), respectively. Confidence equals the winning score, and all three probabilities remain available in the output. Scores are not rounded before choosing the winner. No minimum confidence, vote agreement, or multi-box rejection can override this decision. Invalid uploads and service failures remain HTTP errors, not invented predictions. YOLO continues to locate notes; scan only one note at a time.

Default `CLASSIFIER_INFERENCE_MODE: "direct"` reproduces the reference app's single-image inference, using original JPEG/PNG bytes, with WebP decoded to RGB PNG. Optional `"consensus"` averages the same eight rotation/trim views as V15, then chooses the largest mean probability without any rejection threshold. Direct mode is faster but may be more sensitive to cropping and rotation. Neither mode guarantees correctness; removing abstention can increase incorrect REAL/FAKE decisions. No retraining or model accuracy claim is made.

Existing history is preserved. Old review-policy rows are labelled RESCAN/Legacy scan, not reclassified without running inference again. The new scanner has no UNCERTAIN output. The confidence breakdown still displays the three trained classes. Very close scores can look tied after display rounding.

## Run on Windows

Stop your old project's containers (`docker compose down`, never `down -v` if preserving data), open the extracted BankNoteGuard folder in VS Code, start Docker Desktop and run:

```powershell
docker compose config --quiet
docker compose up --build -d --force-recreate
docker compose logs -f ai
```

Open http://localhost:5174. Check http://localhost:8001/health for ready=true, tensorflowVersion, classifierInferenceMode=direct, decisionRule=argmax, fullImageMinConfidence=null. To check installed TensorFlow directly:

```powershell
docker compose exec ai python -c "import tensorflow as tf; print(tf.__version__); print(tf.keras.backend.backend())"
```

The runtime is installed during Docker build. No separate Windows TensorFlow installation is required. To change the optional inference mode, edit compose.yaml and run `docker compose up -d --build --force-recreate ai`.

## Verification

92 Python regression tests passed; 3 skipped (TensorFlow unavailable in the editing environment). Six frontend confidence tests passed. The Python tests use mocked prediction outputs for decision and API checks; these are not accuracy tests. Model checksum, ZIP integrity and YAML parsing checked. Full Docker, TensorFlow inference, Java and React production builds not run here. See older version notes only for historical context.
