import numpy as np
from numpy.random import *

import torch
from torch.autograd import Variable
from torch.nn.modules import BatchNorm1d, BatchNorm2d, BatchNorm3d

def weights_init(m):
    classname = m.__class__.name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.fine('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)