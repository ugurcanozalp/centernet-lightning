
from typing import Union, Dict, List, Tuple
import time 

import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

from centernet.onnx_inference import ObjectDetector

COLOR_MAP = [(255, 0, 0), (0, 255, 0)]

cap = cv2.VideoCapture("./data/challenge/images/test/test.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = ObjectDetector("deployments/centernet_resnet18.onnx")
cmap = plt.get_cmap("hsv")

times = []
frames = []
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame and bounding boxes
    batch_ids, boxes, scores, labels = detector([frame_rgb])
    for batch_id, box, score, label in zip(batch_ids, boxes, scores, labels):
        x1, y1, x2, y2 = box
        label_name = detector.class_names[label]
        color = COLOR_MAP[label]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        text_to_put = f"{label_name}: {score:.3f}"
        (wtext, htext), _ = cv2.getTextSize(
                text_to_put, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Prints the text.    
        frame = cv2.rectangle(frame, (x1, y1 - 20), (x1 + wtext, y1), color, -1)
        frame = cv2.putText(frame, text_to_put, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
    cv2.imshow('frame', frame)
    times.append(time.time())
    frames.append(frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print("Saving GIF file")
with imageio.get_writer("images/demo.gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        if idx%5==0:
            print("Adding frame to GIF file: ", idx + 1)
            writer.append_data(cv2.cvtColor(cv2.resize(frame, (320, 320)), cv2.COLOR_BGR2RGB))

times = np.array(times)
avg_fps = 1./np.diff(times).mean()
print(f"Average FPS: {avg_fps}")