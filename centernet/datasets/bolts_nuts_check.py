
import torch as th

import json
import numpy as np
import cv2 as cv

cap = cv.VideoCapture("../data/challenge/images/train/train.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

with open("../data/challenge/annotations/instances_train.json") as f:
    annotations = json.load(f)

frame_to_info = {image["id"]: [] for image in annotations["images"]}
for obj in annotations["annotations"]:
    id = obj["image_id"]
    frame_to_info[id].append(obj)

frames_rgb = []
category_color_map = {1: (255, 0, 0), 2: (0, 255, 0)}
text_color = (128, 128, 128)
label_to_text = {1: "nut", 2: "bolt"}
frame_id = 0
while True:
    frame_id += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Display the resulting frame and bounding boxes
    for obj in frame_to_info[frame_id]:
        x, y, w, h = obj["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        label = obj["category_id"]
        label_name = label_to_text[label]
        color = category_color_map[label]
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (wtext, htext), _ = cv.getTextSize(
                label_name, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Prints the text.    
        frame = cv.rectangle(frame, (x, y - 20), (x + wtext, y), color, -1)
        frame = cv.putText(frame, label_name, (x, y - 5),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        # For printing text
        # frame = cv.putText(frame, 'test', (x1, y1),
        #                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv.waitKey(5000)
    cv.imshow('frame', frame)
    print(frame.shape)
    # print(frame)
    if cv.waitKey(1) == ord('q'):
        break
    frames_rgb.append(frames_rgb)
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
