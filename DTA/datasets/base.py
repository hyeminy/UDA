import abc
import random

import torch.utils.data as data
from torchvision import transforms as transforms

from augmentations.misc import Identity


class AbstractDataset(object):

    @staticmethod
    @abc.abstractmethod
    def num_class():
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def code():
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def statistics():
        raise NotImplementedError

    @classmethod
    def train_transform_config(cls, transform_type = None): #cls = VisdaSource class
        config = cls._add_additional_transform(
            {
                'visda_standart_source': transforms.Compoes([
                    transforms.RandomResizedCrop(size224)
                ])
            }
        )# default transform을 의미
        
        if transform_type:
            return config[transform_type] # 특정 transform 반환
        else:
            return config # 전제 transform 반환




    @classmethod
    def eval_transform_config(cls, transforms_type=None):
        pass

    @classmethod
    def _add_additional_transform(cls, default_transform):
        new_transform_config = {}
        for tra

    @classmethod
    def _preprocess_transform(cls):
        pass

    @classmethod
    def _postprocess_transform(cls):
        pass


