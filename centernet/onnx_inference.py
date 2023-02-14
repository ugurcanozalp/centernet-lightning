
from typing import Union, Dict, List, Tuple

import numpy as np
import cv2
import onnxruntime as ort


class ObjectDetector():
    """Object detection inference class.
    """
    def __init__(self, onnx_path: str = "deployments/centernet_resnet18.onnx"):
        self.sess = ort.InferenceSession(onnx_path)
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.class_names = meta["class_names"].split("\n")
        self.stride = int(meta["stride"])
        self.input_shape = np.array([int(meta["input_height"]), int(meta["input_width"])])

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray]:
        """Function to infer objects given in the BGR image. 

        Args:
            image (np.ndarray): 2D Image arrays values between [0-1] with RGB format. 

        Returns:
            batch_ids (th.Tensor): Given an index, it specifies batch index of the detected object. 
            boxes (th.Tensor): Object bounding boxes in (x1, y1, x2, y2) format over input dimensions. 
            scores (th.Tensor): Prediction score of the object class. 
            batch_ids (th.Tensor): Detected object class. 
        """
        image_input = cv2.resize(image, self.input_shape)
        batch = np.expand_dims(image_input.transpose(2, 0, 1), (0, )).astype(np.float32) / 255.0
        batch_ids, boxes, scores, labels = self.sess.run(None, {"image": batch})
        ratio = self.stride * np.array(image.shape[:2]) / self.input_shape
        boxes = (ratio[[1, 0, 1, 0]] * boxes).astype(int)
        return batch_ids, boxes, scores, labels
