
from typing import Optional, Callable, Dict, Tuple, List
import math 

import torch as th
from torchvision import transforms
from torchvision.datasets import CocoDetection

from .utils import gaussian_radius, draw_gaussian


class BoltsNutsCenternet(CocoDetection):
    """Dataset class for Bolts and Nuts dataset with COCO format to train CenterNet. 
    """

    def __init__(self, root: str, annFile: str, phase: str = "train", input_height: int = 160, input_width: int = 160, stride: int = 4) -> None:
        super().__init__(root, annFile)
        self._input_height = input_height
        self._input_width = input_width
        self._stride = stride
        self._output_height = input_height // stride
        self._output_width = input_width // stride

        self.name_to_label = {label["name"]: i for i, label in enumerate(self.coco.cats.values())}
        self.label_to_name = [label["name"] for label in self.coco.cats.values()]
        self._num_classes = len(self.coco.cats)
        if phase != "train":
            self._transforms = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((self._input_height, self._input_width)), 
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
            ])
        else:
            self._transforms = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((self._input_height, self._input_width)), 
            ])            

    def __getitem__(self, i: int) -> Tuple[th.Tensor]:
        image, annotation = super().__getitem__(i)
        bboxes = list()
        labels = list()
        for t in annotation:
            original_height, original_width = self.coco.imgs[annotation[0]["image_id"]]["height"], self.coco.imgs[annotation[0]["image_id"]]["width"]
            ratio_h, ratio_w = self._input_height / original_height, self._input_width / original_width
            x, y, w, h = t["bbox"]
            if min(w, h) <= 0:
                # skip target if width or height is 0
                continue
            bboxes.append(
                [float(x)*ratio_w, float(y)*ratio_h, float(x + w)*ratio_w, float(y + h)*ratio_h]
            )
            labels.append(
                self.name_to_label[self.coco.cats[t["category_id"]]["name"]]
            )
        data = dict(
            image=image,
            bboxes=bboxes,
            labels=labels
        )
        if self._transforms:
            data["image"] = self._transforms(data["image"])
        
        image = data["image"]
        mask = th.zeros(self._num_classes, self._output_height, self._output_width, dtype=th.bool)
        cls_heatmap = th.zeros(self._num_classes, self._output_height, self._output_width, dtype=th.float32)
        bbox_hw = th.zeros(2, self._output_height, self._output_width, dtype=th.float32)
        offset = th.zeros(2, self._output_height, self._output_width, dtype=th.float32)
        
        for bbox, label in zip(data["bboxes"], data["labels"]):
            cy = (bbox[1] + bbox[3]) / 2.0 / self._stride
            cx = (bbox[0] + bbox[2]) / 2.0 / self._stride
            h = (bbox[3] - bbox[1]) / self._stride
            w = (bbox[2] - bbox[0]) / self._stride
            cix = int(cx) 
            ciy = int(cy) 
            hi = math.ceil(h) 
            wi = math.ceil(w) 
            ox = cx - cix 
            oy = cy - ciy 
            mask[label, ciy, cix] = True
            #cls_heatmap[label, ciy, cix] = 1.0
            radius = math.ceil(gaussian_radius((hi, wi)))
            cls_heatmap[label] = draw_gaussian(cls_heatmap[label], (cix, ciy), radius)
            bbox_hw[0, ciy, cix] = h
            bbox_hw[1, ciy, cix] = w
            offset[0, ciy, cix] = oy
            offset[1, ciy, cix] = ox
        return image, mask, cls_heatmap, bbox_hw, offset

if __name__=="__main__":
    ds = BoltsNutsCenternet("../data/challenge/images/train", "../data/challenge/annotations/instances_train.json", phase="train")
    image, mask, cls_heatmap, bbox_hw, offset = ds[325]
    import matplotlib.pyplot as plt 
    plt.imshow(image.permute(1,2,0)); plt.show()
    plt.imshow(cls_heatmap[0]); plt.show()
    plt.imshow(cls_heatmap[1]); plt.show()