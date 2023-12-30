from ultralytics import YOLO 
import cv2

import numpy as np
import matplotlib.pyplot as plt



model = YOLO("/Users/syntichemusawu/Downloads/Gun Detection 2.v1i.yolov8/best (2).pt")

model.predict("/Users/syntichemusawu/Downloads/Gun Detection 2.v1i.yolov8/test/images" , save = True ) #spider

# results = model.predict(source="gunvideo.mp4", stream=True)

source="gunvideo.mp4"
results = model(source, show=True)


cv2.waitKey(0) 