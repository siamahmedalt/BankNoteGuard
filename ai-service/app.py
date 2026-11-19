"""BankNoteGuard AI: single-class YOLO banknote detector + EfficientNet-B0 classifier."""
from __future__ import annotations

import hashlib
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
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("BANKNOTE_MODEL_PATH", str(BASE / "models/best_efficientnetb0.pkl")))
YOLO_PATH = Path(os.getenv("YOLO_WEIGHTS_PATH", str(BASE / "models/banknotes_yolo.pt")))
STORAGE = Path(os.getenv("STORAGE_DIR", str(BASE / "storage")))
DETECT_CONF = float(os.getenv("YOLO_CONFIDENCE", "0.60"))
MIN_NOTE_AREA = float(os.getenv("MIN_NOTE_AREA_RATIO", "0.08"))
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

app = FastAPI(title="BankNoteGuard AI Service", version="2.2.0", lifespan=lifespan)


class PlainDataUnpickler(pickle.Unpickler):
    """Load only plain data from the supplied inference bundle."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Executable types are forbidden in the model bundle")


def load_bundle_metadata():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"EfficientNet bundle missing: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as stream:
        data = PlainDataUnpickler(stream).load()
    if data.get("bundle_format") != "efficientnetb0_banknote_inference_v1":
        raise ValueError("Unsupported EfficientNet bundle format")
    if data.get("class_to_label") != {"real_notes": 0, "fake_notes": 1}:
        raise ValueError("Unexpected EfficientNet class labels")
    if data.get("positive_label") != 1:
        raise ValueError("Expected positive label fake_notes = 1")
    if data.get("preprocessing", {}).get("pixel_range") != "0_to_255":
        raise ValueError("Unexpected EfficientNet preprocessing")
    model_bytes = data.get("keras_model_bytes")
    if not isinstance(model_bytes, bytes) or hashlib.sha256(model_bytes).hexdigest() != data.get("keras_sha256"):
        raise ValueError("Keras checkpoint checksum mismatch")
    return data


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
    """Send only the cropped note to TensorFlow's isolated loopback-only worker."""
    buffer = io.BytesIO()
    cropped.convert('RGB').save(buffer, format='PNG')
    request = Request(WORKER_URL + '/classify', data=buffer.getvalue(),
                      headers={'Content-Type': 'image/png'}, method='POST')
    try:
        with urlopen(request, timeout=90) as response:
            result = json.load(response)
    except (OSError, ValueError, TimeoutError) as exc:
        raise RuntimeError('EfficientNet worker inference unavailable: ' + str(exc)) from exc
    if result.get('status') not in ('REAL', 'FAKE'):
        raise ValueError('Invalid classifier worker result')
    return result


def detect_note(img: Image.Image):
    detector = get_detector()
    names = normalize_names(detector.names)
    mode = detector_mode_from_names(detector.names)
    rgb = np.asarray(img.convert("RGB"))
    prediction = detector.predict(source=rgb, conf=DETECT_CONF, device="cpu", verbose=False)[0]
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
    if not found:
        return None, "No banknote was detected. Try a clear image with the full note visible."
    if len(found) > 1:
        return None, "More than one banknote was detected. Scan one note at a time."
    return found[0], None


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


def _process_image_locked(img: Image.Image, request_id: str, source: str, started: float) -> dict:
    detection, reason = detect_note(img)
    response = {
        "requestId": request_id,
        "source": source,
        "status": "REJECTED",
        "denomination": None,
        "confidence": None,
        "fakeProbability": None,
        "threshold": None,
        "detectorConfidence": None,
        "reason": reason,
        "durationMs": 0,
    }
    if detection:
        cropped = img.crop(detection["bbox"])
        response.update(classify_note(cropped))
        response["denomination"] = detection["denomination"]
        response["detectorConfidence"] = round(detection["confidence"], 6)
        response["reason"] = None
        cropped_dir = STORAGE / "cropped"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        cropped.save(cropped_dir / f"{request_id}.jpg", format="JPEG", quality=90)
    response["durationMs"] = round((time.perf_counter() - started) * 1000)
    archive_result(response)
    LOG.info(
        "SCAN id=%s source=%s status=%s denomination=%s confidence=%s detector_conf=%s elapsed=%sms",
        request_id, source, response["status"], response["denomination"],
        response["confidence"], response["detectorConfidence"], response["durationMs"]
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
    status.update({
        "uptimeSeconds": round(time.time() - START_TIME),
        "classifierBundlePresent": MODEL_PATH.is_file(),
        "yoloWeightsPresent": YOLO_PATH.is_file(),
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
