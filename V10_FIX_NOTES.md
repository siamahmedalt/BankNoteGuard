# V10 rotation and mild crop consistency (experimental)

Based on the user's uploaded `BankNoteGuard_v9.zip`. No changes to trained weights, class mapping, React or Java API schema. Original website features retained.

## Decision policy
- Always classify 0°, 90°, 180°, 270° of the original RGB image and the same four orientations of a symmetric center trim (5% from each edge, rounded down). Every view uses the existing configurable NEAREST resizing by default. YOLO remains localization only, not the source of authenticity crops. Optional multi-note rejection remains disabled by default.
- Average all eight three-class softmax vectors. Accept the highest mean class only if its mean is at least the configured 0.70 threshold and it wins at least 3 out of 4 views in BOTH families. Otherwise return review_uncertain (UI shows UNCERTAIN). REAL and FAKE use exactly the same rule; no forced labels, cached decisions or sample matching. An accepted non-banknote class maps to INVALID. Mean confidence is an uncalibrated model score, not probability of physical authenticity.
- Exact lossless quarter-turns produce the same multiset of image views, modulo numerical inference variation. Arbitrary rotations/resampling, EXIF changes, lossy JPEG, backgrounds and larger crops can still change predictions. Missing physical security features cannot be reconstructed. Eight sequential classifier calls may be slower than V9; no new runtime dependency.

## Tests and known limitations
- 66 Python automated tests passed, including mocked API, rotation-closed pixel-set tests at odd sizes, symmetric decision and vote order checks. These are not model accuracy tests.
- Actual unchanged Keras checkpoint (PyTorch backend, CPU, not deployed TensorFlow): 20 user-labelled REAL originals, 90-degree lossless rotations and symmetric 5% crop on each edge. Raw single-view argmax consistency: 11/20 rotation, 17/20 crop. V10 policy consistency: 20/20 rotation, 19/20 crop. Original V10 outputs: 8 REAL, 1 FAKE, 11 UNCERTAIN. Cropped outputs: 8 REAL, 2 FAKE, 10 UNCERTAIN. One prior screenshot-derived user-labelled FAKE sample became UNCERTAIN (mean fake score 0.562381). See diagnostics/v10_transform_results.csv and diagnostics/v10_fake_screenshot_result.json. Rotation result reuses identical resized inputs to avoid redundant computation. These are classifier-only results without YOLO, not end-to-end tests or independent calibration. User labels were not verified; some images depict out-of-scope 50/100/200 taka denominations. The model can still be wrong even when consistent. Increased abstention is a tradeoff, not an accuracy improvement claim.
- Docker, TensorFlow production runtime, Java and React production build not run here. Existing model metadata does not document training resize interpolation; NEAREST retained from V9. No retraining or threshold tuning performed. For permanent improvement, use original training notebook and independently verified representative genuine/counterfeit/non-note images, augment both classes with rotation and realistic crops, group by physical note before splitting, and evaluate false-FAKE rate, false-REAL rate, abstention and stability on independent groups. Restrict to supported denominations unless trained/validated otherwise. Do not treat image-only output as proof of authenticity.

## Run and diagnose
Stop the old project from its original directory with `docker compose down`. Extract this ZIP and open BankNoteGuard. Run `docker compose up --build -d --force-recreate`. Open http://localhost:5174. Check http://localhost:8001/health for `inferencePolicy: rotation-closed-eight-view-consensus-v10`. View `docker compose logs -f ai` for VIEW and CONSENSUS logs.

To compare the same note at four exact orientations and a mild crop using the running production classifier (no history added):

```
docker compose cp "C:\path\to\note.jpg" ai:/tmp/note.jpg
docker compose exec ai python check_stability.py /tmp/note.jpg
```
