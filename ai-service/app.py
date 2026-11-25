"""BankNoteGuard AI: single-class YOLO banknote detector + EfficientNet-B0 classifier."""
from __future__ import annotations

from contextlib import asynccontextmanager
import io
import logging
import json
import subprocess
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import os
import pickle
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from denomination import recognize_denomination

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("BANKNOTE_MODEL_PATH", str(BASE / "models/best_efficientnetb0_3class.keras")))
YOLO_PATH = Path(os.getenv("YOLO_WEIGHTS_PATH", str(BASE / "models/banknotes_yolo.pt")))
STORAGE = Path(os.getenv("STORAGE_DIR", str(BASE / "storage")))
DETECT_CONF = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
MIN_NOTE_AREA = float(os.getenv("MIN_NOTE_AREA_RATIO", "0.08"))
FULL_IMAGE_MIN_CONF = float(os.getenv("FULL_IMAGE_MIN_CONFIDENCE", "0.70"))
RESIZE_METHOD = os.getenv("CLASSIFIER_RESIZE_METHOD", "NEAREST").upper()
if RESIZE_METHOD not in {"NEAREST", "LANCZOS"}:
    raise ValueError("CLASSIFIER_RESIZE_METHOD must be NEAREST or LANCZOS")
if not 0.5 <= FULL_IMAGE_MIN_CONF <= 1.0:
    raise ValueError("FULL_IMAGE_MIN_CONFIDENCE must be between 0.5 and 1.0")
VERIFY_MULTIPLE_NOTES = os.getenv("VERIFY_MULTIPLE_NOTES", "false").lower() in {"true", "1", "yes"}
MULTI_BOX_MIN_CONF = float(os.getenv("MULTI_BOX_MIN_CONFIDENCE", "0.90"))
MULTI_CLASS_MIN_CONF = float(os.getenv("MULTI_CLASS_MIN_CONFIDENCE", "0.85"))
MAX_SIZE = 10 * 1024 * 1024
DENOMINATION_NAMES = {"500_taka": 500, "1000_taka": 1000}
SINGLE_CLASS_NAME = "banknote"
VALID_TYPES = {"image/jpeg", "image/png", "image/webp"}
ARCHIVE_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
START_TIME = time.time()
WORKER_URL = 'http://127.0.0.1:8002'
WORKER_PROCESS = None

MODEL_STATE = {
    "service": "online", "ready": False, "loading": True,
    "message": "Waiting for YOLO and EfficientNet initialization",
    "error": None, "detectorMode": None, "detectorClasses": [],
    "classifierReady": False,
}

LOG = logging.getLogger("banknoteguard.ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

def _worker_health():
    try:
        with urlopen(f"{WORKER_URL}/health", timeout=3) as reply:
            return json.load(reply)
    except (OSError, ValueError, TimeoutError) as exc:
        return {"ready": False, "error": str(exc), "message": "EfficientNet worker unavailable"}


def _bootstrap_models():
    global WORKER_PROCESS
    with STATE_LOCK:
        MODEL_STATE.update(loading=True, ready=False, error=None,
                           message="Loading YOLO and separate EfficientNet worker")
    LOG.info("Starting isolated model processes: YOLO stays in API; TensorFlow runs in local worker")
    try:
        # This child is isolated from PyTorch native libraries loaded by this API process.
        WORKER_PROCESS = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'classifier_worker:app',
            '--host', '127.0.0.1', '--port', '8002', '--workers', '1'
        ], cwd=str(BASE), stdout=None, stderr=None)
        detector = get_detector()
        names = normalize_names(detector.names)
        mode = detector_mode_from_names(detector.names)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if WORKER_PROCESS.poll() is not None:
                raise RuntimeError(f"EfficientNet worker exited (code {WORKER_PROCESS.returncode})")
            state = _worker_health()
            if state.get('ready'):
                break
            if state.get('error') and state.get('loading') is False:
                raise RuntimeError('EfficientNet worker failed: ' + str(state['error']))
            time.sleep(1)
        else:
            raise TimeoutError('EfficientNet worker did not become ready in 180 seconds')
        with STATE_LOCK:
            MODEL_STATE.update(loading=False, ready=True, error=None,
                               classifierReady=True, detectorMode=mode,
                               detectorClasses=[names[n] for n in sorted(names)],
                               message="YOLO and isolated EfficientNet worker ready")
        LOG.info("ALL MODELS READY: YOLO=%s; classifier worker PID=%s", mode, WORKER_PROCESS.pid)
    except Exception as exc:
        LOG.exception('AI model initialization failed')
        with STATE_LOCK:
            MODEL_STATE.update(loading=False, ready=False, classifierReady=False,
                               error=f"{type(exc).__name__}: {exc}",
                               message=f"AI model initialization failed: {type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_bootstrap_models, daemon=True, name="ai-model-loader").start()
    yield
    if WORKER_PROCESS is not None and WORKER_PROCESS.poll() is None:
        WORKER_PROCESS.terminate()
        try:
            WORKER_PROCESS.wait(timeout=8)
        except subprocess.TimeoutExpired:
            WORKER_PROCESS.kill()

app = FastAPI(title="BankNoteGuard AI Service", version="8.0.0", lifespan=lifespan)


def normalize_names(names):
    """Return YOLO class names as {integer class id: class name}."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise ValueError(f"Unsupported YOLO class-name format: {type(names).__name__}")


def detector_mode_from_names(names):
    """Accept this uploaded one-class checkpoint or a future two-denomination detector."""
    values = set(normalize_names(names).values())
    if values == {SINGLE_CLASS_NAME}:
        return "single-class"
    if values == set(DENOMINATION_NAMES):
        return "denomination"
    raise ValueError(
        "Unsupported YOLO classes. This project accepts {'banknote'} or "
        "{'500_taka', '1000_taka'}; received " + repr(sorted(values))
    )


def get_classifier():
    """Backward-compatible readiness accessor; TensorFlow never loads in this process."""
    state = _worker_health()
    if not state.get('ready'):
        raise RuntimeError('EfficientNet worker not ready: ' + str(state.get('error') or state.get('message')))
    return state


@lru_cache(maxsize=1)
def get_detector():
    if not YOLO_PATH.is_file():
        raise FileNotFoundError(f"YOLO weights missing: {YOLO_PATH}")
    from ultralytics import YOLO

    detector = YOLO(str(YOLO_PATH))
    mode = detector_mode_from_names(detector.names)
    LOG.info("YOLO detector ready; mode=%s; classes=%s", mode, normalize_names(detector.names))
    return detector


def classify_note(cropped: Image.Image):
    """Send a resized crop or full image to the isolated classifier."""
    buffer = io.BytesIO()
    cropped.convert('RGB').resize((224, 224), Image.Resampling[RESIZE_METHOD]).save(buffer, format='PNG')
    request = Request(WORKER_URL + '/classify', data=buffer.getvalue(),
                      headers={'Content-Type': 'image/png'}, method='POST')
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (OSError, ValueError, TimeoutError) as exc:
        raise RuntimeError('EfficientNet worker inference unavailable: ' + str(exc)) from exc
    if result.get('status') not in ('REAL', 'FAKE', 'REJECTED'):
        raise ValueError('Invalid classifier worker result')
    return result


def detect_candidates(img: Image.Image):
    detector = get_detector()
    names = normalize_names(detector.names)
    mode = detector_mode_from_names(detector.names)
    # Ultralytics treats NumPy image inputs as BGR, unlike Pillow RGB.
    bgr = np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])
    prediction = detector.predict(source=bgr, conf=DETECT_CONF, iou=0.45, agnostic_nms=True,
                                  max_det=30, device="cpu", verbose=False)[0]
    found = []
    if prediction.boxes is not None:
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            class_name = names.get(class_id)
            if mode == "single-class" and class_name == SINGLE_CLASS_NAME:
                denomination = None
            elif mode == "denomination" and class_name in DENOMINATION_NAMES:
                denomination = DENOMINATION_NAMES[class_name]
            else:
                continue
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / (img.width * img.height)
            if area_ratio < MIN_NOTE_AREA:
                continue
            found.append({
                "denomination": denomination,
                "className": class_name,
                "confidence": float(box.conf.item()),
                "bbox": (x1, y1, x2, y2),
            })
    return found


def overlap(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    inter = max(0, min(x2, u2)-max(x1, u1))*max(0, min(y2, v2)-max(y1, v1))
    aa, bb = (x2-x1)*(y2-y1), (u2-u1)*(v2-v1)
    return inter/(aa+bb-inter), inter/min(aa,bb)


def consolidate_candidates(candidates):
    """Collapse duplicate/contained detections, retaining the best enclosing box."""
    ordered = sorted(candidates, key=lambda d: d['confidence'], reverse=True)
    kept = []
    for candidate in ordered:
        duplicates = [k for k in kept if overlap(k['bbox'], candidate['bbox'])[0] >= .45
                      or overlap(k['bbox'], candidate['bbox'])[1] >= .80]
        if not duplicates:
            kept.append(dict(candidate))
            continue
        # Union avoids losing security features when one box covers only part of a note.
        # A later box may bridge two prior boxes; collapse both rather than count them twice.
        group = duplicates + [candidate]
        for k in duplicates:
            kept.remove(k)
        box = (min(k['bbox'][0] for k in group), min(k['bbox'][1] for k in group),
               max(k['bbox'][2] for k in group), max(k['bbox'][3] for k in group))
        best = max(group, key=lambda k:k['confidence'])
        denominations = {k['denomination'] for k in group}
        kept.append({**best, 'bbox': box, 'denomination': best['denomination'] if len(denominations)==1 else None})
    return sorted(kept, key=lambda d:d['confidence'], reverse=True)


def crop_note(img, bbox):
    x1,y1,x2,y2 = bbox
    # Include a small border so edge features are not cut off.
    px, py = round((x2-x1)*.04), round((y2-y1)*.04)
    return img.crop((max(0,x1-px),max(0,y1-py),min(img.width,x2+px),min(img.height,y2+py)))


def plausible_whole_note(candidate):
    x1,y1,x2,y2 = candidate['bbox']
    w,h = x2-x1,y2-y1
    return candidate['confidence'] >= MULTI_BOX_MIN_CONF and 1.5 <= max(w,h)/min(w,h) <= 3.2


def confident_note(result, threshold):
    confidence = result.get('confidence')
    return (result.get('status') in ('REAL', 'FAKE')
            and isinstance(confidence, (int, float))
            and threshold <= confidence <= 1.0)


def confirmed_multiple(img, candidates):
    """Only reject for multiple notes when distinct whole-note-like crops agree.

    Box count alone cannot tell fragments of one note from separate notes.
    Restrict verification to three plausible boxes to bound CPU inference latency.
    """
    strong = [c for c in candidates if plausible_whole_note(c)][:3]
    verified = []
    for c in strong:
        if not any(overlap(c['bbox'], other['bbox'])[1] <= .10 for other in strong if other is not c):
            continue
        result = classify_note(crop_note(img,c['bbox']))
        if confident_note(result, MULTI_CLASS_MIN_CONF):
            if any(overlap(c['bbox'], other['bbox'])[1] <= .10 for other in verified):
                return True
            verified.append(c)
    return False


def archive_result(result: dict):
    """Append plain prediction metadata; never modify either trained model."""
    target = STORAGE / "prediction_results.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE_LOCK, target.open("ab") as stream:
        pickle.dump(result, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())


def process_image(img: Image.Image, request_id: str, source: str) -> dict:
    started = time.perf_counter()
    with INFERENCE_LOCK:
        return _process_image_locked(img, request_id, source, started)


def verify_prediction(img, candidates, request_id):
    """Conservative multi-view inference: never select a label by maximum confidence.

    Full-image REAL remains primary; FAKE and uncertain decisions need corroboration.
    Alternate resampling is not independent evidence of authenticity. The policy is
    provisional until tested on a representative, independent banknote set.
    """
    views = []
    def predict_view(name, image):
        result = classify_note(image)
        views.append((name, result))
        LOG.info("VIEW id=%s name=%s status=%s confidence=%s probabilities=%s",
                 request_id, name, result['status'], result.get('confidence'),
                 result.get('probabilities'))
        return result

    primary = predict_view('original_' + RESIZE_METHOD.lower(), img)
    threshold = FULL_IMAGE_MIN_CONF
    # Preserve confident original REAL results. Do not rotate square close-ups.
    if primary['status'] == 'REAL' and confident_note(primary, threshold):
        return primary, 'full_image_primary', views

    # A second resize can expose a fragile result without modifying source content.
    predict_view('original_bilinear', img.resize((224, 224), Image.Resampling.BILINEAR))
    # A third view uses a detector crop only when it covers most of the upload.
    # Otherwise use bicubic interpolation of the original, not an arbitrary fragment.
    if len(candidates) == 1:
        x1, y1, x2, y2 = candidates[0]['bbox']
        coverage = (x2-x1)*(y2-y1)/(img.width*img.height)
    else:
        coverage = 0.0
    if coverage >= .55 and coverage < .96:
        predict_view('large_detector_crop', crop_note(img, candidates[0]['bbox']))
    else:
        predict_view('original_bicubic', img.resize((224,224), Image.Resampling.BICUBIC))

    # Only inspect quarter-turns for clearly portrait-shaped notes. Both directions
    # are tested; do not cherry-pick the one with the largest score. Partial square
    # crops keep their original orientation.
    if img.height / img.width >= 1.65 and primary['status'] != 'REJECTED':
        predict_view('rotate_90', img.transpose(Image.Transpose.ROTATE_90))
        predict_view('rotate_270', img.transpose(Image.Transpose.ROTATE_270))

    original_views = views[:3]
    strong = {label: [r for _, r in original_views if r['status'] == label
                      and (r.get('confidence') or 0) >= threshold]
              for label in ('REAL', 'FAKE', 'REJECTED')}
    # Any strong competing class makes a definitive result unsafe.
    # A strong non-note result is never rescued by an isolated crop.
    if (primary['status'] == 'REJECTED' and (primary.get('confidence') or 0) >= threshold
            and len(strong['REJECTED']) >= 2 and not strong['REAL'] and not strong['FAKE']):
        return strong['REJECTED'][0], 'non_banknote_confirmed', views
    for label, opposite in [('REAL','FAKE'), ('FAKE','REAL')]:
        if (primary['status'] == label and confident_note(primary, threshold)
                and len(strong[label]) >= 2 and not strong[opposite] and not strong['REJECTED']):
            rotated = [r for _, r in views[3:]]
            if len(rotated) == 2 and all(r['status'] == opposite and confident_note(r, threshold) for r in rotated):
                break
            # Use the mean confidence of agreeing predictions, not the maximum.
            # Keep the actual mean FAKE softmax score for audit.
            agreeing = strong[label]
            result = dict(agreeing[0])
            result['confidence'] = round(sum(r['confidence'] for r in agreeing)/len(agreeing), 6)
            fake_scores = [r.get('fakeProbability') for r in agreeing]
            if all(v is not None for v in fake_scores):
                result['fakeProbability'] = round(sum(fake_scores)/len(fake_scores), 6)
            result.pop('probabilities', None)
            return result, 'multi_view_verified', views
    return dict(status='REJECTED', confidence=None, fakeProbability=primary.get('fakeProbability'),
                reason='Prediction uncertain or conflicting. Rescan in good lighting; authenticity is not established.'), 'review_uncertain', views


def _process_image_locked(img: Image.Image, request_id: str, source: str, started: float) -> dict:
    raw = detect_candidates(img)
    candidates = consolidate_candidates(raw)
    response = {
        "requestId": request_id, "source": source, "status": "REJECTED",
        "denomination": None, "confidence": None, "fakeProbability": None,
        "threshold": None,
        "detectorConfidence": max((round(c['confidence'], 6) for c in candidates), default=None),
        "reason": None, "durationMs": 0, "inferencePath": "full_image_primary",
        "detectorBoxCount": len(raw), "candidateCount": len(candidates),
    }
    selected_image = img
    if len(candidates) > 1 and VERIFY_MULTIPLE_NOTES and confirmed_multiple(img, candidates):
        response.update(reason="Multiple separate banknotes detected. Scan one note at a time.",
                        inferencePath="multiple_confirmed")
    else:
        result, path, views = verify_prediction(img, candidates, request_id)
        # Keep the Java response schema unchanged; detailed view probabilities go into logs.
        response.update({k:v for k,v in result.items() if k != 'probabilities'},
                        threshold=FULL_IMAGE_MIN_CONF, inferencePath=path)
        if response['status'] != 'REJECTED' and len(candidates) == 1:
            response['denomination'] = candidates[0]['denomination']
            selected_image = crop_note(img, candidates[0]['bbox'])

    # Finalize ALL inference paths, not just detector_crop.
    if response['status'] == 'REJECTED':
        response['denomination'] = None
        response['reason'] = response['reason'] or 'Image is not a supported banknote.'
    else:
        response['reason'] = None
        if response['denomination'] is None:
            try:
                amount = recognize_denomination(selected_image, img)
                response['denomination'] = amount if amount in (500, 1000) else None
            except Exception:
                # OCR is optional; its failure must not erase authenticity.
                LOG.exception('Denomination OCR failed for %s', request_id)
            LOG.info('DENOMINATION id=%s source=ocr amount=%s', request_id, response['denomination'])
        cropped_dir = STORAGE / 'cropped'
        cropped_dir.mkdir(parents=True, exist_ok=True)
        selected_image.convert('RGB').save(cropped_dir / f'{request_id}.jpg', format='JPEG', quality=90)

    LOG.info('DETECTION id=%s raw_boxes=%s consolidated=%s path=%s',
             request_id, len(raw), len(candidates), response['inferencePath'])
    response['durationMs'] = round((time.perf_counter() - started) * 1000)
    archive_result(response)
    LOG.info(
        "SCAN id=%s source=%s status=%s denomination=%s confidence=%s detector_conf=%s elapsed=%sms",
        request_id, source, response['status'], response['denomination'],
        response['confidence'], response['detectorConfidence'], response['durationMs'],
    )
    return response


@app.get("/health/live")
def health_live():
    return {"service": "online", "live": True}


@app.get("/health")
@app.get("/health/ready")
def health():
    with STATE_LOCK:
        status = dict(MODEL_STATE)
    if status.get('ready'):
        worker = _worker_health()
        if not worker.get('ready') or (WORKER_PROCESS is not None and WORKER_PROCESS.poll() is not None):
            status.update(ready=False, classifierReady=False, loading=False,
                          message='EfficientNet worker disconnected; check AI logs',
                          error=worker.get('error') or 'Worker process exited')
    if status.get('ready'):
        status['classifierModelSha256'] = worker.get('modelSha256')
        status['tensorflowVersion'] = worker.get('tensorflowVersion')
    status.update({
        "uptimeSeconds": round(time.time() - START_TIME),
        "classifierBundlePresent": MODEL_PATH.is_file(),
        "yoloWeightsPresent": YOLO_PATH.is_file(),
        "inferencePolicy": "nearest-primary-v8-conservative-multiview-review",
        "fullImageMinConfidence": FULL_IMAGE_MIN_CONF,
        "classifierResizeMethod": RESIZE_METHOD,
        "verifyMultipleNotes": VERIFY_MULTIPLE_NOTES,
    })
    return status


@app.post("/predict")
def predict(image: UploadFile = File(...), source: str = Form("upload"), request_id: str = Form("")):
    if source not in {"camera", "upload"}:
        raise HTTPException(400, "source must be camera or upload")
    if image.content_type not in VALID_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG, or WebP image")
    with STATE_LOCK:
        ready = MODEL_STATE["ready"]
        message = MODEL_STATE["message"]
    if not ready:
        raise HTTPException(503, message)
    if not _worker_health().get('ready'):
        raise HTTPException(503, 'EfficientNet worker is temporarily unavailable')
    raw = image.file.read(MAX_SIZE + 1)
    if len(raw) > MAX_SIZE:
        raise HTTPException(413, "Maximum image size is 10 MB")
    if not raw:
        raise HTTPException(400, "The uploaded image is empty")
    try:
        scan_id = str(uuid.UUID(request_id)) if request_id else str(uuid.uuid4())
    except ValueError:
        raise HTTPException(400, "Invalid request ID")
    try:
        with Image.open(io.BytesIO(raw)) as incoming:
            if incoming.width * incoming.height > 25_000_000:
                raise HTTPException(413, "Image resolution exceeds 25 megapixels")
            incoming.verify()
        with Image.open(io.BytesIO(raw)) as incoming:
            corrected = ImageOps.exif_transpose(incoming).convert("RGB")
            corrected.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "The file is not a valid image")
    uploads = STORAGE / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    corrected.save(uploads / f"{scan_id}.jpg", format="JPEG", quality=92)
    try:
        return process_image(corrected, scan_id, source)
    except (FileNotFoundError, ValueError) as error:
        LOG.error("Model configuration error: %s", error)
        raise HTTPException(503, str(error))
    except Exception:
        LOG.exception("Inference failed for %s", scan_id)
        raise HTTPException(500, "Model inference failed; inspect the AI service log")


@app.get('/history-image/{scan_id}')
def history_image(scan_id: uuid.UUID):
    path = STORAGE / 'uploads' / f'{scan_id}.jpg'
    if not path.is_file():
        raise HTTPException(404, 'Image unavailable')
    with Image.open(path) as image:
        image.thumbnail((240, 160))
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG', quality=85)
    return Response(buffer.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})
