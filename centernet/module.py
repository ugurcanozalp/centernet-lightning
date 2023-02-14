
from typing import Union, Dict, List, Tuple
from argparse import ArgumentParser
import math 

import torch as th
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl

from .archs import backbone_map


class BinaryFocalLoss(nn.Module):
    """Loss function over heatmap defined on Eq.1 in the paper. 
    https://arxiv.org/abs/1904.07850
    """

    def __init__(self, alpha: float, beta: float):
        super(BinaryFocalLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta

    def forward(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        """Function to calculate loss given prediction and target class heatmaps. 

        Args:
            pred (th.Tensor): Predicted heatmap with shape (batch_size, num_class, output_height, output_width)
            target (th.Tensor): Target heatmap with shape (batch_size, num_class, output_height, output_width)

        Returns:
            th.Tensor: Loss value
        """
        one_target = - (1.0 - pred).pow(self._alpha) * pred.log() # if target is 1
        zero_target = - pred.pow(self._alpha) * (1.0 - pred).log() # if target is 0
        loss = th.where(target == 1.0, one_target, (1.0 - target).pow(self._beta) * zero_target)
        return loss.mean()


class CenterNet(pl.LightningModule):
    """Generic pl.LightningModule definition for CenterNet object detection. 
    https://arxiv.org/abs/1904.07850
    """

    def __init__(self, 
        backbone: str = "resnet18", 
        class_names: List[str] = ["a", "b"], 
        num_classes: int = 2, 
        input_height: int = 160, 
        input_width: int = 160,
        threshold: float = 0.5, 
        learning_rate: float = 1e-4, 
        alpha: float = 2.0, 
        beta: float = 4.0, 
        **kwargs
    ):
        super(CenterNet, self).__init__()
        self.save_hyperparameters()
        self._metrics = {
            "map": MeanAveragePrecision()
        }
        self._backbone = backbone_map[backbone]()
        self.stride = self._backbone.stride
        self._output_normalizer = math.sqrt(input_height * input_width) / self.stride
        self._embedding_size = self._backbone.embedding_size
        self._cls_head = nn.Sequential(
            nn.Conv2d(self._embedding_size, self._embedding_size, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(self._embedding_size, self.hparams.num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self._cls_loss = BinaryFocalLoss(self.hparams.alpha, self.hparams.beta)
        self._bbox_hw_head = nn.Sequential(
            nn.Conv2d(self._embedding_size, self._embedding_size, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(self._embedding_size, 2, kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )
        self._bbox_hw_loss = nn.L1Loss()
        self._offset_head = nn.Sequential(
            nn.Conv2d(self._embedding_size, self._embedding_size, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(self._embedding_size, 2, kernel_size=1, stride=1, padding=0),
        )
        self._offset_loss = nn.L1Loss()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((self.hparams.input_height, self.hparams.input_width))
        ])

    def forward(self, image: th.Tensor) -> Tuple[th.Tensor]:
        """Main inference function for object detection.

        Args:
            image (th.Tensor): Input tensor with shape (batch_size, 3, input_height, input_width)

        Returns:
            batch_ids (th.Tensor): Given an index, it specifies batch index of the detected object. 
            boxes (th.Tensor): Object bounding boxes in (x1, y1, x2, y2) format over output heatmap. 
            scores (th.Tensor): Prediction score of the object class. 
            batch_ids (th.Tensor): Detected object class. 
        """
        feature_map = self._backbone(image) # batch, embedding, height/stride, width/stride
        cls_map = self._cls_head(feature_map) # batch, num_classes, height/stride, width/stride
        bbox_hw_map = self._bbox_hw_head(feature_map) # batch, 2, height/stride, width/stride
        offset_map = self._offset_head(feature_map) # batch, 2, height/stride, width/stride
        batch_ids, boxes, scores, labels = self._postprocess(cls_map, bbox_hw_map, offset_map)
        return batch_ids, boxes, scores, labels

    def _postprocess(self, cls_map: th.Tensor, bbox_hw_map: th.Tensor, offset_map: th.Tensor, mask_map: Union[None, th.Tensor]=None) -> Tuple[th.Tensor]:
        """Postprocess function to detect objects from heatmap features.

        Args:
            cls_map (th.Tensor): Class heatmap produced by network.
            bbox_hw_map (th.Tensor): Height and width for each point on output map no matter there is object or not.
            offset_map (th.Tensor): Offset for height and width for each point on output map no matter there is object or not.
            mask_map (Union[None, th.Tensor], optional): Mask to decide which points to predict objects. Only used when target data is passed. Defaults to None.

        Returns:
            batch_ids (th.Tensor): Given an index, it specifies batch index of the detected object. 
            boxes (th.Tensor): Object bounding boxes in (x1, y1, x2, y2) format over output heatmap. 
            scores (th.Tensor): Prediction score of the object class. 
            batch_ids (th.Tensor): Detected object class. 
        """
        if mask_map is None:
            diff_mask_map = F.max_pool2d(cls_map, 3, stride=1, padding=1) >= cls_map # batch, num_classes, height/stride, width/stride
            threshold_mask_map = cls_map > self.hparams.threshold
            mask_map = th.logical_and(diff_mask_map, threshold_mask_map)
        batch_ids, labels, height_idx, width_idx = th.where(mask_map)
        scores = cls_map[batch_ids, labels, height_idx, width_idx]
        bbox_hws = bbox_hw_map[batch_ids, :, height_idx, width_idx]
        offsets = offset_map[batch_ids, :, height_idx, width_idx]
        y1 = height_idx.float() + offsets[:, 0] - bbox_hws[:, 0] / 2.0
        y2 = height_idx.float() + offsets[:, 0] + bbox_hws[:, 0] / 2.0
        x1 = width_idx.float() + offsets[:, 1] - bbox_hws[:, 1] / 2.0
        x2 = width_idx.float() + offsets[:, 1] + bbox_hws[:, 1] / 2.0
        boxes = th.stack([x1, y1, x2, y2], dim=-1) # num_detected, 4
        return batch_ids, boxes, scores, labels

    def configure_optimizers(self) -> th.optim.Optimizer:
        optimizer = th.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        lr_scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 40])
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: th.Tensor, batch_idx: int) -> Dict[str, th.Tensor]:
        (image, mask_map_target, cls_map_target, bbox_hw_map_target, offset_map_target),  = batch
        feature_map = self._backbone(image)
        cls_map = self._cls_head(feature_map)
        bbox_hw_map = self._bbox_hw_head(feature_map)
        offset_map = self._offset_head(feature_map) 
        cls_loss = self._cls_loss(cls_map, cls_map_target)
        loss_mask = mask_map_target.any(axis=1) # batch_size, output_height, output_width
        bbox_hw_loss = self._bbox_hw_loss(bbox_hw_map.permute(0, 2, 3, 1)[loss_mask], bbox_hw_map_target.permute(0, 2, 3, 1)[loss_mask]) / self._output_normalizer
        offset_loss = self._offset_loss(offset_map.permute(0, 2, 3, 1)[loss_mask], offset_map_target.permute(0, 2, 3, 1)[loss_mask]) / self._output_normalizer
        loss = 10*cls_loss + bbox_hw_loss + offset_loss
        info = {
            "loss": loss, 
            "cls_loss": cls_loss, 
            "bbox_loss": bbox_hw_loss, 
            "offset_loss": offset_loss
        }
        for key, value in info.items():
            self.log(key, value)
        return info

    def validation_step(self, batch: th.Tensor, batch_idx: int) -> Dict:
        image, mask_map_target, cls_map_target, bbox_hw_map_target, offset_map_target = batch
        batch_size = image.shape[0]
        batch_ids_target, boxes_target, scores_target, labels_target = self._postprocess(cls_map_target, bbox_hw_map_target, offset_map_target, mask_map=mask_map_target)
        batch_ids, boxes, scores, labels = self.forward(image)
        preds = [{"boxes": boxes[batch_ids==i], "scores": scores[batch_ids==i], "labels": labels[batch_ids==i]} for i in range(batch_size)]
        targets = [{"boxes": boxes_target[batch_ids_target==i], "labels": labels_target[batch_ids_target==i]} for i in range(batch_size)]
        self._metrics["map"].update(preds, targets)
        return {}

    def test_step(self, batch: th.Tensor, batch_idx: int) -> Dict:
        image, mask_map_target, cls_map_target, bbox_hw_map_target, offset_map_target = batch
        batch_size = image.shape[0]
        batch_ids_target, boxes_target, scores_target, labels_target = self._postprocess(cls_map_target, bbox_hw_map_target, offset_map_target, mask_map=mask_map_target)
        batch_ids, boxes, scores, labels = self.forward(image)
        preds = [{"boxes": boxes[batch_ids==i], "scores": scores[batch_ids==i], "labels": labels[batch_ids==i]} for i in range(batch_size)]
        targets = [{"boxes": boxes_target[batch_ids_target==i], "labels": labels_target[batch_ids_target==i]} for i in range(batch_size)]
        self._metrics["map"].update(preds, targets)
        return {}

    def training_epoch_end(self, outputs: List[Dict[str, th.Tensor]]) -> None:
        pass

    def validation_epoch_end(self, outputs: List[Dict[str, th.Tensor]]) -> None:
        map_metrics = self._metrics["map"].compute()
        for name, value in map_metrics.items():
            self.log(name, value)

    def test_epoch_end(self, outputs: List[Dict[str, th.Tensor]]) -> None:
        map_metrics = self._metrics["map"].compute()
        for name, value in map_metrics.items():
            self.log(name, value)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--backbone", type=str, default="resnet18")
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--input_height", type=int, default=160)
        parser.add_argument("--input_width", type=int, default=160)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--alpha", type=float, default=2.0)
        parser.add_argument("--beta", type=float, default=4.0)
        return parser