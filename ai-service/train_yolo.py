"""Requires your OWN annotated image dataset. Example:
python train_yolo.py --data datasets/banknotes.yaml --epochs 80
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='datasets/banknotes.yaml')
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--imgsz', type=int, default=640)
args = parser.parse_args()
config = Path(args.data).resolve()
if not config.is_file():
    raise SystemExit(f'Create dataset YAML and labeled images first: {config}')
model = YOLO('yolo11n.pt')  # general COCO pretrained starting point ONLY; must fine-tune
result = model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz, project='runs', name='banknotes')
print('Training finished. Validate with: yolo detect val model=runs/banknotes/weights/best.pt data=' + str(config))
print('Then COPY runs/banknotes/weights/best.pt -> models/banknotes_yolo.pt')
