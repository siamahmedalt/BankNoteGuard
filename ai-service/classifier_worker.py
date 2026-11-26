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
RESIZE_METHOD = os.getenv("CLASSIFIER_RESIZE_METHOD", "BILINEAR").upper()
if RESIZE_METHOD != "BILINEAR":
    raise ValueError("CLASSIFIER_RESIZE_METHOD must be BILINEAR to match training")
STATE = {'ready': False, 'loading': True, 'error': None}
MODEL = None
BUNDLE = None
LOCK = threading.Lock()

def load_bundle():
    mapping = json.loads((MODEL_PATH.parent / 'class_mapping_3class.json').read_text())
    if mapping != {'real_notes': 0, 'fake_notes': 1, 'not_banknotes': 2}:
        raise ValueError('Unexpected classifier labels')
    contract = json.loads((MODEL_PATH.parent / 'inference_contract.json').read_text())
    if (contract['class_mapping'] != mapping or contract['input_size'] != [224, 224, 3]
            or contract['resize_method'] != 'BILINEAR' or contract['antialias'] is not False
            or contract['dtype'] != 'float32' or contract['pixel_range'] != [0, 255]):
        raise ValueError('Invalid training/inference contract')
    actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if actual_hash != contract['model_sha256']:
        raise ValueError('Classifier checksum mismatch; install matching model and inference contract')
    return {'class_to_label': mapping, 'preprocessing': {'image_size': [224, 224]}, 'model_sha256': actual_hash}


def preprocess_image(raw):
    """Same floating-point resize and pixel scale as notebook load_image()."""
    import tensorflow as tf
    # Reference mode sends original JPEG/PNG bytes; consensus sends RGB PNG views.
    # Validate dimensions before TensorFlow allocates tensors.
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in ('PNG', 'JPEG') or image.width * image.height > 25_000_000:
            raise ValueError('Expected JPEG or PNG with at most 25 megapixels')
        image.verify()
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [224, 224], method='bilinear', antialias=False)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)


# Backward-compatible name used by existing tests.
preprocess_png = preprocess_image


def decode_prediction(prediction):
    probabilities = np.asarray(prediction, dtype=float).ravel()
    if len(probabilities) != 3 or not np.all(np.isfinite(probabilities)) or not np.isclose(probabilities.sum(), 1, atol=1e-3) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Invalid three-class softmax output')
    index = int(np.argmax(probabilities))
    return {'status': ('REAL', 'FAKE', 'REJECTED')[index],
            'confidence': float(probabilities[index]),
            'fakeProbability': float(probabilities[1]),
            'probabilities': {name: float(probabilities[i]) for i, name in enumerate(('real_notes','fake_notes','not_banknotes'))},
            'threshold': None,
            'reason': 'Image is not a supported banknote.' if index == 2 else None}

def initialize():
    global MODEL, BUNDLE
    try:
        import tensorflow as tf  # Never import PyTorch/Ultralytics in this process
        if tf.keras.backend.backend() != "tensorflow":
            raise RuntimeError("TensorFlow backend is required")
        BUNDLE = load_bundle()
        MODEL = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=True)
        if tuple(MODEL.input_shape[1:]) != (224, 224, 3) or MODEL.output_shape[-1] != 3:
            raise ValueError('Unexpected classifier input/output shape')
        decode_prediction(MODEL.predict(np.zeros((1,224,224,3), dtype=np.float32), verbose=0))
        STATE.update(ready=True, loading=False, error=None, modelSha256=BUNDLE['model_sha256'], tensorflowVersion=tf.__version__, classifierResizeMethod=RESIZE_METHOD)
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
    # Native RGB PNG can exceed the compressed upload size (25 MP = 75 MB RGB).
    if not raw or len(raw) > 100*1024*1024:
        raise HTTPException(413, 'Classification image exceeds allowed size')
    try:
        pixels = preprocess_image(raw)
    except (OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(400, 'Invalid classification image')
    with LOCK:
        prediction = MODEL.predict(pixels, verbose=0)
    try:
        return decode_prediction(prediction)
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc
