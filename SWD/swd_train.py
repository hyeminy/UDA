from __future__ import print_function

from swd_options import config

import torch
import torch as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms

import numpy as np
import os

from utils import *
from taskcv_loader import CVDataLoader
from basenet import *


def train(config):

    use_gpu = torch.cuda.is_available()

    torch.manual_seed(config['seed'])

    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])


    save_path = config['save_path']

    source_path = config['source_path']
    target_path = config['target_path']

    data_transforms = {
        source_path : transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),

        target_path : transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    }

    dsets = {
        source_path : datasets.ImageFolder(os.path.join(source_path),
                                           data_transforms[source_path]),
        target_path : datasets.ImageFolder(os.path.join(target_path),
                                           data_transforms[target_path])
    }

    train_loader = CVDataLoader()




    num_k = config['num_k'] # step 3 반복 횟수

    num_layer = config['num_layer']

    batch_size = config['batch_size']





if __name__ == "__main__":
    train(config)