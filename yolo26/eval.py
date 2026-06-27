from ultralytics import YOLO
from compress.Compress import PruneHandler as ph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_pt', type=str, default='../test_model.pt')
parser.add_argument('--model_yaml', type=str, default=None)

# parser.add_argument('--model_pt', type=str, default='/home/a/capstone/model/g040_i048.pt')
# parser.add_argument('--model_yaml', type=str, default='/home/a/capstone/model/g040_i048.yaml')

parser.add_argument('--data', type=str, default="../dataset.yaml")
parser.add_argument('--split', type=str, default="test", choices=["train", "val", "test"])
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()
if args.model_yaml:
    model = YOLO(args.model_yaml).load(args.model_pt)
else:
    model = YOLO(args.model_pt)
model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
