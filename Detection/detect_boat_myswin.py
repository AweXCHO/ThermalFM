import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import sys
import os
# sys.path.append('/mnt/dqdisk/Code/dinov3/dinov3')
# sys.path.insert(0, '/mnt/dqdisk/Code/dinov3')
# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()
# 预测框粗细和颜色修改问题可看<新手推荐学习视频.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    # model = YOLO('./ultralytics/cfg/models/11/yolov11n-dinov3-obb.yaml')
    model = YOLO(model = r'/mnt/dqdisk/Code/ultralytics-yolo11-main/runs/train/myswin_boat/exp3/weights/best_fp32.pt')
    #model.load('/mnt/dqdisk/Code/ultralytics-yolo11-main/runs/train/dota/exp9/weights/best.pt') # select your model.pt path
    model.predict(source='/mnt/dqdisk/Data/Boat_coco/images/val',
                  imgsz=448,
                  rect=False,
                  project='inference/myswin_boat',
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