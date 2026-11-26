# V14 — Scan confidence display

Based on V13, without changing the model, input preprocessing, decision rules or thresholds. The scanner displays the actual averaged softmax scores from the same eight views used by the existing decision policy.

- REAL, FAKE and INVALID (non-banknote) display their prediction confidence and the three-class probability breakdown.
- UNCERTAIN displays the highest model score with its class label, the breakdown, and an active review-rule indicator. UNCERTAIN is not a fourth model class; this score is not confidence that a note is uncertain. An uncertain decision can occur even when the highest score exceeds 70%, if view agreement fails.
- If no final classification was performed (for example, optional multiple-note rejection), no class confidence is fabricated: unavailable values display as a dash.
- The AI API exposes `realProbability`, `fakeProbability`, `invalidProbability` in the 0–1 range. The Java backend validates the breakdown and persists the two newly added nullable fields. Existing `confidence` semantics and history table remain unchanged; historical records are not retroactively inferred. Existing PostgreSQL schema update configuration will add the two nullable columns on startup.
- Model scores are not calibrated guarantees of physical authenticity.

## Run

Stop the previous project using `docker compose down` from its folder (do not use `-v` if preserving database history). Extract this ZIP, open the BankNoteGuard folder and run `docker compose up --build -d --force-recreate`. Rebuild all services, because both AI and Java response contracts and the React scanner have changed. Open http://localhost:5174 and scan a new image. If keeping existing named volumes, use the same Compose project name as before. Browser refresh may be necessary.

## Checks performed

- `cd ai-service && python -m pytest -q`: 75 passed, 1 skipped (TensorFlow numerical parity check, TensorFlow unavailable).
- `cd frontend && npm test`: 6 passed (confidence formatting, all four displayed decisions, missing scores).
- JSX syntax checked with TypeScript transpilation; no diagnostics.
- Existing model weights unchanged; ZIP integrity and YAML/XML checked.
- Java/Maven compilation, full React production build and Docker integration not performed in this environment. Regression tests use mocked inference, not new accuracy measurements.
