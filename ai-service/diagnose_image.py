"""Inspect actual uploaded file with the running classifier (no model retraining)."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
import app as ai

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        size = image.size
        orientation = image.getexif().get(274, 1)
        corrected = ImageOps.exif_transpose(image).convert('RGB')
        corrected.load()
    health = ai._worker_health()
    if not health.get('ready'):
        raise SystemExit(f'Classifier not ready: {health}')
    print(json.dumps({'file': str(args.image), 'file_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
                      'raw_size': size, 'exif_orientation': orientation, 'corrected_size': corrected.size,
                      'model_sha256': hashlib.sha256(ai.MODEL_PATH.read_bytes()).hexdigest(),
                      'worker': health, 'threshold': ai.FULL_IMAGE_MIN_CONF}, indent=2))
    result, path, views = ai.verify_prediction(corrected, [], 'diagnostic')
    print(json.dumps({'result_without_detector_crop': result, 'path': path, 'views': views}, indent=2))
    print('Diagnostic omits YOLO crop and does not add website history. Compare with VIEW logs for the actual website scan.')

if __name__ == '__main__':
    main()
