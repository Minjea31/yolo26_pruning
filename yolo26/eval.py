import sys
from ultralytics import YOLO
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='../test_model.pt')
    parser.add_argument('--data', type=str, default="../dataset.yaml")

    args = parser.parse_args()

    model = YOLO(args.model)

    metrics = model.val(data=args.data)
    print('mAP50:', metrics.box.map50)
