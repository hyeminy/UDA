from __future__ import print_function

from swd_options import config

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms

import os

from utils import *
from taskcv_loader import CVDataLoader

from basenet import *

def train(config):
    #####################################################
    # ENV
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(config['seed'])

    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])

    save_path = config['save_path']
    #######################################################

    source_path = config['source_path']
    target_path = config['target_path']

    num_k = config['num_k']

    num_layer =








if __name __ == "__main__":
    train(config)