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
                      'worker': health, 'decision_rule': 'argmax'}, indent=2))
    with Image.open(args.image) as source:
        fmt = source.format
    original = (args.image.read_bytes(), 'image/jpeg' if fmt == 'JPEG' else 'image/png') if fmt in ('JPEG','PNG') else None
    result, path, views = ai.classify_for_policy(corrected, [], 'diagnostic', original)
    direct, direct_path, raw = ai.direct_prediction(corrected, 'reference-comparison', original)
    print(json.dumps({'reference_raw_argmax':raw[0][1], 'direct_argmax':direct, 'direct_path':direct_path}, indent=2))
    print(json.dumps({'result_without_detector_crop': result, 'path': path, 'views': views}, indent=2))
    print('Diagnostic uses the configured authenticity policy as the website, without history. Compare with DIRECT and AVERAGE logs.')

if __name__ == '__main__':
    main()
