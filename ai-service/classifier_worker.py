"""Isolated TensorFlow EfficientNet classifier. Only listens on localhost:8002."""
from __future__ import annotations
import hashlib
import io
import logging
import os
import pickle
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError

LOG = logging.getLogger('banknoteguard.classifier')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')
MODEL_PATH = Path(os.getenv('BANKNOTE_MODEL_PATH', '/app/models/best_efficientnetb0.pkl'))
STATE = {'ready': False, 'loading': True, 'error': None}
MODEL = None
BUNDLE = None
LOCK = threading.Lock()

class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('Executable pickle globals are forbidden')

def load_bundle():
    with MODEL_PATH.open('rb') as f:
        bundle = DataOnlyUnpickler(f).load()
    if not isinstance(bundle, dict) or bundle.get('bundle_format') != 'efficientnetb0_banknote_inference_v1':
        raise ValueError('Unsupported classifier model bundle')
    if bundle.get('class_to_label') != {'real_notes': 0, 'fake_notes': 1}:
        raise ValueError('Unexpected classifier labels')
    if bundle.get('preprocessing', {}).get('pixel_range') != '0_to_255':
        raise ValueError('Unexpected pixel preprocessing')
    raw = bundle.get('keras_model_bytes')
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != bundle.get('keras_sha256'):
        raise ValueError('Classifier checksum mismatch')
    return bundle

def initialize():
    global MODEL, BUNDLE
    try:
        import tensorflow as tf  # Never import PyTorch/Ultralytics in this process
        BUNDLE = load_bundle()
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / 'classifier.keras'
            model_path.write_bytes(BUNDLE['keras_model_bytes'])
            MODEL = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=True)
        height, width = BUNDLE['preprocessing']['image_size']
        prediction = np.asarray(MODEL.predict(np.zeros((1,height,width,3),dtype=np.float32),verbose=0)).ravel()
        if len(prediction) != 1 or not np.isfinite(prediction[0]):
            raise ValueError('Classifier warmup failed')
        STATE.update(ready=True, loading=False, error=None)
        LOG.info('EfficientNet isolated worker READY; threshold=%.6f', BUNDLE['decision_threshold'])
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
        raise HTTPException(413, 'Crop exceeds allowed size')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            height, width = BUNDLE['preprocessing']['image_size']
            resized = image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
            pixels = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
    except (OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(400, 'Invalid crop')
    with LOCK:
        probability = float(np.asarray(MODEL.predict(pixels, verbose=0)).ravel()[0])
    threshold = float(BUNDLE['decision_threshold'])
    if not np.isfinite(probability):
        raise HTTPException(500, 'Classifier prediction invalid')
    fake = probability >= threshold
    return {'status': 'FAKE' if fake else 'REAL', 'fakeProbability': round(probability, 6),
            'confidence': round(probability if fake else 1-probability, 6),
            'threshold': round(threshold,6)}
