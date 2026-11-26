"""Compare rotations and crops using the deployed classifier, without adding scan history."""
import argparse
import json
from PIL import Image, ImageOps
import app as ai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    args = parser.parse_args()
    if not ai._worker_health().get('ready'):
        raise SystemExit('Classifier worker not ready; check docker compose logs ai')
    with Image.open(args.image) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
    dx, dy = image.width//20, image.height//20
    variants = [('original', image)]
    variants.extend((f'rotate_{n}', image.transpose(turn)) for n,turn in
                    [(90,Image.Transpose.ROTATE_90),(180,Image.Transpose.ROTATE_180),(270,Image.Transpose.ROTATE_270)])
    variants.append(('crop_5pct_each_edge', image.crop((dx,dy,image.width-dx,image.height-dy))))
    results = []
    for name, im in variants:
        result, path, _ = ai.classify_for_policy(im, [], name)
        status = 'INVALID' if result['status'] == 'REJECTED' else result['status']
        results.append(dict(variant=name, status=status, path=path, confidence=result['confidence'], probabilities=result['probabilities']))
    print(json.dumps({'results':results,'quarter_turns_consistent':len({r['status'] for r in results[:4]})==1,'mild_crop_consistent':results[0]['status']==results[-1]['status']},indent=2))

if __name__ == '__main__':
    main()
