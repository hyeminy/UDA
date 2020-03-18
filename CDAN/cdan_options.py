import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import lr_schedule
import network
import loss
import pre_process as prep
import random
import pdb
import math

parser = argparse.ArgumentParser(description='Conditional Damain Adversarial Network')

parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--net', type=str, default='ResNet50',
                    choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16",
                             "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                    help="The dataset or source dataset used")
parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt',
                    help="The source dataset path list")
parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt',
                    help="The target dataset path list")
parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
parser.add_argument('--output_dir', type=str, default='san',
                    help="output directory of our model (in ../snapshot directory)")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--random', type=bool, default=False, help="whether use random projection")

##################
# 아래 부분은 원래 코드에는 없었는데 내가 추가
parser.add_argument('--num_iterations', type=int, default=100004)
parser.add_argument('--output_for_test', type=bool, default=True)
parser.add_argument('--class_num', type=int, default=31)
parser.add_argument('--trade_off', type=float, default=1.0)
parser.add_argument('--source_batchsize',type=int, default=36)
parser.add_argument('--target_batchsize',type=int, default=36)
parser.add_argument('--test_batchsize',type=int, default=4)
##################

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# config에 따로 parser 값을 다 넣어 줌
config = {}
config['method'] = args.method
config['gpu_id'] = args.gpu_id
config['num_iterations'] = args.num_iterations
config['test_interval'] = args.test_interval
config['snapshot_interval'] = args.snapshot_interval
config['output_for_test'] = args.output_for_test

config['output_path'] = args.output_dir
if not os.path.exists(config["output_path"]):
    os.mkdir(config["output_path"])

config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w') #log를 다 저장
if not os.path.exists(config["output_path"]):
    os.mkdir(config["output_path"])

##########
#내가 추가한 부분
config['s_dset_path']=args.s_dset_path
config['t_dset_path']=args.t_dset_path
##########


# CDAN은 test를 이상하게 함 --> 이 부분 수정해야 해 (MDD 참고하기)
config["prep"] = {'test_10crop': True, \
                  'params': {'resize_size': 256, \
                             'crop_size': 224,
                             }
                  }

config["loss"] = {"trade_off": args.trade_off}  # 원래 1.0으로 fix되어있는데, args.trade_off로 수정
config['loss']['random'] = args.random  # 어디에 쓰이는 거지
config['loss']['random_dim'] = 1024

# optimizer 부분도 option 다양하게 args로 넘기도록 수정 필요
# lr도 여러 option 추가해서 넣을 순 없을까?
config['optimizer'] = {'type': optim.SGD,
                       'optim_params': {'lr': args.lr,
                                        'momentum': 0.9,
                                        'weight_decay': 0.0005,
                                        'nesterov': True},
                       'lr_type': "inv",
                       'lr_param': {"lr": args.lr,
                                    'gamma': 0.001,
                                    'power': 0.75
                                    }
                       }
config['dataset'] = args.dset
config['data'] = {"source": {"list_path": args.s_dset_path,
                             "batch_size": args.source_batchsize},
                  "target": {"list_path": args.t_dset_path,
                             "batch_size": args.target_batchsize},
                  'test': {"list_path": args.t_dset_path,
                           "batch_size": args.test_batchsize}
                  }
#batch size 원래 fix였는데 수정함


if "ResNet" in args.net:
    config['network'] = {'name': network.ResNetFC,
                         'params': {'resnet_name': args.net,
                                    'use_bottleneck': True,
                                    'bottleneck_dim': 256,
                                    'new_cls': True,
                                    'class_num': args.class_num}
                         }
else:  # 다른 네트워크 추가하려면 이 부분 추가해야 함
    pass


config['out_file'].write(str(config))
config['out_file'].flush()


print('----------------------------------------')
for key, value in config.items():
    print(key, ":", value)
print('----------------------------------------')