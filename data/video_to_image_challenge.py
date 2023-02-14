
import os
import cv2
import numpy as np

video_files = ["challenge/images/train/train.mp4", "challenge/images/val/val.mp4", "challenge/images/test/test.mp4"]

for video_file in video_files:
    path = os.path.dirname(video_file)
    cap = cv2.VideoCapture(video_file)
    id = 0
    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        filename = os.path.join(path, "{id:04d}.jpg".format(id=id))
        cv2.imwrite(filename, frame)
        id += 1
