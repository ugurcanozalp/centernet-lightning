
import os
import tempfile
from argparse import ArgumentParser

import torch as th
import pytorch_lightning as pl
import onnx
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType

from centernet import CenterNet


parser = ArgumentParser()
parser.add_argument('--ckpt', type=str, default="checkpoints/centernet_resnet18.pt.ckpt")
parser.add_argument("--target-path", "-t", type=str, default="deployments/", help="target path to save model")
parser.add_argument("--quantize", "-q", action="store_true", help="int8 quantization")
parser = CenterNet.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)

model = CenterNet.load_from_checkpoint(args.ckpt)

if not os.path.exists(args.target_path):
    os.mkdir(args.target_path)

# onnx export configs
opset_version = 17

dynamic_axes = {
    "image": {0: "batch", 2: "height", 3: "width"},  # write axis names
    "batch_ids": {0: "num_obj"}, 
    "boxes": {0: "num_obj"}, 
    "scores": {0: "num_obj"}, 
    "labels": {0: "num_obj"}
}

input_names = [
    "image",
]

output_names = [
    "batch_ids", 
    "boxes", 
    "scores", 
    "labels"
]

# define dummy sample
input_sample = th.rand(1, 3, model.hparams.input_height, model.hparams.input_width)

# export model as onnx
if args.quantize:
    target_model_path = os.path.join(
        args.target_path, "centernet_{}_quantized.onnx".format(args.backbone))
    with tempfile.NamedTemporaryFile(suffix=".onnx") as foo:
        model.to_onnx(foo.name,
                      input_sample=input_sample,
                      opset_version=opset_version,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      export_params=True)
        quantize_dynamic(foo.name, target_model_path, weight_type=QuantType.QUInt8)
else:
    target_model_path = os.path.join(
        args.target_path, "centernet_{}.onnx".format(args.backbone))
    model.to_onnx(target_model_path,
                  input_sample=input_sample,
                  opset_version=opset_version,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes=dynamic_axes,
                  export_params=True)

onnx_model = onnx.load(target_model_path)

meta = onnx_model.metadata_props.add()
meta.key = "class_names"
meta.value = "\n".join(model.hparams.class_names)
meta = onnx_model.metadata_props.add()
meta.key = "stride"
meta.value = str(model.stride)
meta = onnx_model.metadata_props.add()
meta.key = "input_height"
meta.value = str(model.hparams.input_height)
meta = onnx_model.metadata_props.add()
meta.key = "input_width"
meta.value = str(model.hparams.input_width)
onnx_model.doc_string = 'centernet model'
onnx.save(onnx_model, target_model_path)
