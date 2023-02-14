
import os
from argparse import ArgumentParser

import torch as th
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from centernet import CenterNet
from centernet.datasets import BoltsNutsCenternet

parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, default="checkpoints/centernet_resnet18.pt.ckpt")
parser.add_argument('--batch_size', type=int, default=16)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

trainer = pl.Trainer(gpus=args.gpus)

model = CenterNet.load_from_checkpoint(args.ckpt)

test_ds = BoltsNutsCenternet(
    "./data/challenge/images/test", 
    "./data/challenge/annotations/instances_test.json", 
    phase="test", 
    input_height=model.hparams.input_height, 
    input_width=model.hparams.input_width, 
    stride=model.stride)

test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=12)

trainer.test(model, dataloaders=[test_dl])
