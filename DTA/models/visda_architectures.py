import math

import torch.nn as nn
import torchvision.models.resnet as resnet

from models.resnet import Bottleneck, model_urls
from models.base import AbstractModel

def create_resnet_lower(model_name='resnet50', pretrained=True):
    models = {'resnet50' : resnet50}
    return models[model_name](pretrained=pretrained, is_lower=True)

def resnet50(pretrained=True, is_lower=True, num_classes=12, **kwargs):
    if is_lower:
        model = ResNetLower(Bottleneck, [3,4,6,2], **kwargs)


class ResNetLower(AbstractModel):
    def __init__(self, block, layers):
        