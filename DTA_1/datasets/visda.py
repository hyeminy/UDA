import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from augmentations.misc import ToRGB
from datasets.base import AbstractDataSet
from dta_options import args

VISDA_CHANNEL_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
source_path = args.source_path
target_path = args.target_path

class VisdaSource(ImageFolder, AbstractDataSet):
    def __init__(self,root=source_path, **kwargs):
        root = root
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 12

    @staticmethod
    def code():
        return "visda_source"

    @staticmethod
    def statistice():
        return VISDA_CHANNEL_STATS

class VisdaTarget(ImageFolder, AbstractDataSet):
    def __init__(self, root=target_path, **kwargs):
        root = root
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 12

    @staticmethod
    def code():
        return "visda_target"

    @staticmethod
    def statistics():
        return VISDA_CHANNEL_STATS

    @classmethod
    def _preprocess_transform(cls):
        return transforms.Compose([ToRGB()])