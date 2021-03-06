import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from augmentations.misc import ToRGB
from datasets.base import AbstractDataSet
from dta_options import args

amazon_path = args.amazon_path
dslr_path = args.dslr_path
webcam_path = args.webcam_path

OFFICE31_CHANNELS_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Amazon(ImageFolder, AbstractDataSet):
    def __init__(self,root=amazon_path, **kwargs):
        root = root
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 31

    @staticmethod
    def code():
        return "amazon_source"

    @staticmethod
    def statistice():
        return OFFICE31_CHANNELS_STATS


class Dslr(ImageFolder, AbstractDataSet):
    def __init__(self,root=dslr_path, **kwargs):
        root = root
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 31

    @staticmethod
    def code():
        return "dslr_source"

    @staticmethod
    def statistice():
        return OFFICE31_CHANNELS_STATS

class Webcam(ImageFolder, AbstractDataSet):
    def __init__(self,root=webcam_path, **kwargs):
        root = root
        super().__init__(root, **kwargs)

    @staticmethod
    def num_class():
        return 31

    @staticmethod
    def code():
        return "webcam_source"

    @staticmethod
    def statistice():
        return OFFICE31_CHANNELS_STATS