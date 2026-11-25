"""Isolated TensorFlow EfficientNet classifier. Only listens on localhost:8002."""
from __future__ import annotations
import json
import hashlib
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError

LOG = logging.getLogger('banknoteguard.classifier')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')
MODEL_PATH = Path(os.getenv('BANKNOTE_MODEL_PATH', str(Path(__file__).resolve().parent / 'models/best_efficientnetb0_3class.keras')))
RESIZE_METHOD = os.getenv("CLASSIFIER_RESIZE_METHOD", "NEAREST").upper()
if RESIZE_METHOD not in {"NEAREST", "LANCZOS"}:
    raise ValueError("CLASSIFIER_RESIZE_METHOD must be NEAREST or LANCZOS")
STATE = {'ready': False, 'loading': True, 'error': None}
MODEL = None
BUNDLE = None
LOCK = threading.Lock()

def load_bundle():
    mapping = json.loads((MODEL_PATH.parent / 'class_mapping_3class.json').read_text())
    if mapping != {'real_notes': 0, 'fake_notes': 1, 'not_banknotes': 2}:
        raise ValueError('Unexpected classifier labels')
    return {'class_to_label': mapping, 'preprocessing': {'image_size': [224, 224]}}


def decode_prediction(prediction):
    probabilities = np.asarray(prediction, dtype=float).ravel()
    if len(probabilities) != 3 or not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1, atol=1e-3) or np.any(probabilities < 0):
        raise ValueError('Invalid three-class softmax output')
    index = int(np.argmax(probabilities))
    return {'status': ('REAL', 'FAKE', 'REJECTED')[index],
            'confidence': round(float(probabilities[index]), 6),
            'fakeProbability': round(float(probabilities[1]), 6),
            'probabilities': {name: round(float(probabilities[i]), 6) for i, name in enumerate(('real_notes','fake_notes','not_banknotes'))},
            'threshold': None,
            'reason': 'Image is not a supported banknote.' if index == 2 else None}

def initialize():
    global MODEL, BUNDLE
    try:
        import tensorflow as tf  # Never import PyTorch/Ultralytics in this process
        BUNDLE = load_bundle()
        MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=True)
        if tuple(MODEL.input_shape[1:]) != (224, 224, 3) or MODEL.output_shape[-1] != 3:
            raise ValueError('Unexpected classifier input/output shape')
        decode_prediction(MODEL.predict(np.zeros((1,224,224,3), dtype=np.float32), verbose=0))
        STATE.update(ready=True, loading=False, error=None, modelSha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), tensorflowVersion=tf.__version__, classifierResizeMethod=RESIZE_METHOD)
        LOG.info('Three-class EfficientNet worker READY')
    except Exception as exc:
        LOG.exception('EfficientNet worker failed to load')
        STATE.update(ready=False, loading=False, error=str(exc))

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=initialize, daemon=True).start()
    yield

app = FastAPI(title='Isolated BankNoteGuard EfficientNet worker', lifespan=lifespan)

@app.get('/health')
def health():
    return dict(STATE)

@app.post('/classify')
async def classify(request: Request):
    if not STATE['ready']:
        raise HTTPException(503, 'Classifier worker is not ready')
    raw = await request.body()
    if not raw or len(raw) > 12*1024*1024:
        raise HTTPException(413, 'Classification image exceeds allowed size')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            height, width = BUNDLE['preprocessing']['image_size']
            resized = image.convert('RGB').resize((width, height), Image.Resampling[RESIZE_METHOD])
            pixels = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
    except (OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(400, 'Invalid classification image')
    with LOCK:
        prediction = MODEL.predict(pixels, verbose=0)
    try:
        return decode_prediction(prediction)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
