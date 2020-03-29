import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from augmentations.misc import ToRGB
from datasets.base import AbstractDataSet
from dta_options import args


OFFICE31_CHANNELS_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
#source_path = args.source_path
#target_path = args.target_path

class Amazon(ImageFolder, AbstractDataSet):
    def __init__(self,root=source_path, **kwargs):
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
    def __init__(self,root=source_path, **kwargs):
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
    def __init__(self,root=source_path, **kwargs):
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