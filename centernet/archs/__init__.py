
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152

backbone_map = {
	"resnet18": resnet18,
	"resnet34": resnet34,
	"resnet50": resnet50,
	"resnet101": resnet101,
	"resnet152": resnet152,
}
