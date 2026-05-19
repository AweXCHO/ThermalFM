import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import sys
import os


if __name__ == '__main__':
    # model = YOLO('./ultralytics/cfg/models/11/yolov11n-dinov3-obb.yaml')
    model = YOLO(model = r'runs/train/myswin_llvip/exp5/weights/best_fp32.pt')
    #model.load('/mnt/dqdisk/Code/ultralytics-yolo11-main/runs/train/dota/exp9/weights/best.pt') # select your model.pt path
    model.predict(source='/mnt/dqdisk/Data/LLVIP/images/val',
                  imgsz=448,
                  rect=False,
                  project='inference/person/myswin',
                  name='exp',
                  save=True,
                  conf=0.2,
                  iou=0.5,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=2, # line width of the bounding boxes
                  show_conf=False, # do not show prediction confidence
                  show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
