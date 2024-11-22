# Imports

import ultralytics
from ultralytics import YOLO
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("CUDA available:", torch.cuda.is_available())

if __name__ == "__main__":
    model = YOLO('yolov8m.pt')

    result = model.train(
        data='C:\\Users\\Yasins\\Desktop\\UAV\\configs\\config.yaml',
        epochs=100,
        patience=15,
        imgsz=640,
        workers=8,
        batch=16, # When I set 32, I get GPU out of memory
        device=0,
        optimizer="AdamW", # Explicitly set optimizer
        project='C:\\Users\\Yasins\\Desktop\\UAV\\models\\trained',
        name='yolov8m-uav_50k'
    )
