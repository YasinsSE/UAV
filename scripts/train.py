# Imports

import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
from glob import glob
import shutil
import time
import seaborn as sns
from PIL import Image
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":
    model = YOLO('yolov8n.pt')

    model.train(
        data='data\config.yaml',
        epochs=20,
        patience=5,
        imgsz=640,
        workers=8,
        batch=16,
        device=0,
        name='yolov8n-uav'
    )
