
# CenterNet Object Detection
Object detection training/inference pipeline using [CenterNet algorithm](https://arxiv.org/pdf/1904.07850). Training with [Pytorch Lightning](https://www.pytorchlightning.ai/) and inference with [ONNX Runtime](https://www.onnxruntime.ai/). Fow now, only [ResNet](https://arxiv.org/pdf/1512.03385) backbones are available, but others such as [HourGlass](https://arxiv.org/abs/1603.06937) network will be added soon.

## Installation
```
git clone https://github.com/ugurcanozalp/centernet-lightning
cd centernet-lightning
pip install -e .
```
## Training
[Pytorch Lightning](https://www.pytorchlightning.ai/) is used for training. First, place data files into data folder. To train resnet18 model with bolts and nuts dataset, run the following command,
```bash
python train_bolts_nuts.py --gpus 1
```
For testing the model, run the following command.
```bash
python test_bolts_nuts.py --gpus 1
```

## Inference
For inference, you should use one of checkpoints (for below example, 
![checkpoints](/checkpoints) folder)
```python
from centernet import CenterNet
from PIL import Image
model = CenterNet.load_from_checkpoint("checkpoints/centernet_resnet18.pt.ckpt") # Load pretrained model.
image = Image.open("images/test_0336.jpg") 
batch = model.preprocess(image).unsqueeze(0) # convert to specific size and torch tensor, add batch dimension
batch_ids, boxes, scores, labels = model(batch)
```

### Onnx Runtime Inference
If you want to use onnx runtime, export the model using `export.py`. 
```
python -m scripts.export --ckpt checkpoints/centernet_resnet18.pt.ckpt --quantized
```

Then, you do inference as follows.
```python
import numpy as np
from PIL import Image
from centernet import ObjectDetector
image = Image.open("images/test_0336.jpg") 
image = np.asarray(image) # PIL image to numpy array
detector = ObjectDetector("deployments/centernet_resnet18_quantized.onnx")
batch_ids, boxes, scores, labels = detector([image])
```

### ONNX real-time demo
You can use `demo.py` script for this purpose. Modify it according to your purposes.

```
python demo.py
```

Current trained model have following output in test set (bolts and nuts).
![Expected Output](images/demo.gif)

## References
- [Objects as Points](https://arxiv.org/pdf/1911.02116.pdf)

## Citation

```bibtex
@inproceedings{zhou2019objects,
    title={Objects as Points},
    author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
    booktitle={arXiv preprint arXiv:1904.07850},
    year={2019}
}
```

# Author

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ugurcanozalp/)

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@ugurcanozalp)

[![StackOverFlow](https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/11985314/u%c4%9fur-can-%c3%96zalp)
