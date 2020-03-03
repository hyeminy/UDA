import numpy as np
import math
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
from torchvision import models

import transnorm_resnet
from trans_norm import TransNorm1d, TransNorm2d



resnet_dict = {"ResNet50": transnorm_resnet.resnet50}

class ResNetFc(nn.Module):
    def __init__(self, \
                 resnet_name, \
                 use_bottleneck=True, \
                 bottleneck_dim=256, \
                 new_cls=False,\
                 class_num=1000,\
                 type=None):
        network_config = {'type' : type}

        super(ResNetFc, self).__init__()
        
        model_resnet = resnet_dict[resnet_name](pretrained=True, **network_config) # transnorm_resnet.resnet50(pretrained=True)
        print('-------------------------Done----------------------------------')
        print(model_resnet)
        print('---------------------------------------------------------------')
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.feautre_layers = nn.Sequential(self.conv1,\
                                            self.bn1,\
                                            self.relu,\
                                            self.maxpool,\
                                            self.layer1,\
                                            self.layer2,\
                                            self.layer3,\
                                            self.layer4,\
                                            self.avgpool)


        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck_linear = nn.Linear(2048, bottleneck_dim)
                self.bottleneck_linear.weight.data.normal_(0, 0.005)
                self.bottleneck_linear.bias.data.fill_(0.0)
                if type == 'normal':
                    self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim)
                else:
                    self.bottleneck_bn = TransNorm1d(bottleneck_dim)
                #print(self.bottleneck_bn)
                self.bottleneck_relu = nn.ReLU(True)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(2048, class_num)
                self.fc.weight.data.normal_(0,0.01)
                self.fc.bias.data.fill_(0.0)

            self.__in_features = bottleneck_dim

        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, mode='train'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        x = self.bottleneck_linear(x)

        if mode == 'train':
            x, tau, cur_mean_source, cur_mean_target, output_mean_source, output_mean_target = self.bottleneck_bn(x, last_flag=True)

        else:
            x = self.bottleneck_bn(x)

        x = self.bottleneck_relu(x)
        y = self.fc(x)

        if mode == 'train':
            return x, y, tau, cur_mean_source, cur_mean_target, output_mean_source, output_mean_target

        else:
            return x, y

    def output_dim(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [
                    {"params" : self.feautre_layers.parameters(),\
                     "lr_mult" : 1,\
                     'decay_mult' : 2},
                    {"params": self.bottleneck_linear.parameters(), \
                     "lr_mult": 10, \
                     'decay_mult': 2},
                    {"params": self.bottleneck_bn.parameters(), \
                     "lr_mult": 10, \
                     'decay_mult': 2},
                    {"params": self.fc.parameters(), \
                     "lr_mult": 10, \
                     'decay_mult': 2},
                ]

            else:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]

        else:
            parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class AdversarialNetwork(nn.Module):

    def __init__(self, in_feature, hidden_size):  # ( base_network.output_num()*class_num, 1024 )

        super(AdversarialNetwork, self).__init__()

        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.ad_layer3 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 50000.0

    def forward(self, x):

        if self.training:
            self.iter_num +=1

        coeff = calc_coeff(self.iter_num,\
                           self.high,\
                           self.low,\
                           self.alpha,\
                           self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))  # GRL 부분 같은데

        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        y = self.ad_layer3(x)
        y = self.sigmoid(y)

        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params" : self.parameters(),
                 "lr_mult" : 10,
                 "decay_mult" : 2}]



def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1 :
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    elif classname.find('BatchNorm') != -1 :
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

