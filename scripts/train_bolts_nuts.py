
import os
from argparse import ArgumentParser

import torch as th
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from centernet import CenterNet
from centernet.datasets import BoltsNutsCenternet

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser = CenterNet.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)

# define checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="./checkpoints",
    verbose=True,
    filename="centernet_"+args.backbone+".pt",
    monitor="map",
    save_top_k=1,
    save_weights_only=True,
    mode="max" # only pick max of `map`
)

trainer = pl.Trainer(
    gpus=args.gpus,
    callbacks=[checkpoint_callback],
    gradient_clip_val=args.gradient_clip_val, 
    max_epochs=args.max_epochs)

class_names = ["bolt", "nut"]

model = CenterNet(class_names=class_names, **dict_args)

train_ds = BoltsNutsCenternet(
    "./data/challenge/images/train", 
    "./data/challenge/annotations/instances_train.json", 
    phase="train", 
    input_height=args.input_height, 
    input_width=args.input_width, 
    stride=model.stride)

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=12)

val_ds = BoltsNutsCenternet(
    "./data/challenge/images/val", 
    "./data/challenge/annotations/instances_val.json", 
    phase="val", 
    input_height=args.input_height, 
    input_width=args.input_width, 
    stride=model.stride)

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=12)

trainer.fit(model, train_dataloaders=[train_dl], val_dataloaders=[val_dl])

