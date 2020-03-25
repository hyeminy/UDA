from transnorm_options import config

import numpy as np
import pdb
import math
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets

import lr_schedule
import network
import loss
import pre_process as prep
#import random

#from tensorboardX import SummaryWriter


def train(config):
    ####################################################
    # Tensorboard setting
    ####################################################
    #tensor_writer = SummaryWriter(config["tensorboard_path"])

    ####################################################
    # Data setting
    ####################################################

    prep_dict = {} # 데이터 전처리 transforms 부분
    prep_dict["source"] = prep.image_train(**config['prep']['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test"] = prep.image_test(**config['prep']['params'])

    dsets = {}
    dsets["source"]= datasets.ImageFolder(config['s_dset_path'], transform=prep_dict["source"])
    dsets["target"]= datasets.ImageFolder(config['t_dset_path'], transform=prep_dict['target'])
    dsets['test']=datasets.ImageFolder(config['t_dset_path'],transform=prep_dict['test'])


    data_config = config["data"]
    train_source_bs = data_config["source"]["batch_size"]   #원본은 source와 target 모두 source train bs로 설정되었는데 이를 수정함
    train_target_bs = data_config['target']['batch_size']
    test_bs = data_config["test"]["batch_size"]

    dset_loaders = {}
    dset_loaders["source"]=DataLoader(dsets["source"], batch_size=train_source_bs, shuffle=True, num_workers=4, drop_last=True) # 원본은 drop_last=True, 이렇게 해야 마지막까지 source, target에서 동일한 수로 배치 생성가능
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_target_bs, shuffle=True, num_workers=4, drop_last=True)
    dset_loaders['test'] = DataLoader(dsets['test'], batch_size=test_bs, shuffle=False, num_workers=4, drop_last=False)

    ####################################################
    # Network Setting
    ####################################################

    class_num = config["network"]['params']['class_num']

    net_config = config["network"]
    """
        config['network'] = {'name': network.ResNetFc,
                         'params': {'resnet_name': args.net,
                                    'use_bottleneck': True,
                                    'bottleneck_dim': 256,
                                    'new_cls': True,
                                    'class_num': args.class_num,
                                    'type' : args.type}
                         }
    """

    base_network = net_config["name"](**net_config["params"])
    #network.py에 정의된 ResNetFc() 클래스 호출
    base_network = base_network.cuda() # ResNetFc(Resnet, True, 256, True, 12)

    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num],
                                           config["loss"]["random_dim"]
                                           )
        random_layer.cuda()
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num()*class_num, 1024) # 왜 class 수 만큼 곱하지?

    ad_net = ad_net.cuda()

    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ####################################################
    # Env Setting
    ####################################################

    #gpus = config['gpu'].split(',')
    #if len(gpus) > 1 :
        #ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        #base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ####################################################
    # Optimizer Setting
    ####################################################

    optimizer_config = config['optimizer']
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    # optim.SGD
    
    #config['optimizer'] = {'type': optim.SGD,
                           #'optim_params': {'lr': args.lr,
                                            #'momentum': 0.9,
                                            #'weight_decay': 0.0005,
                                            #'nestrov': True},
                           #'lr_type': "inv",
                           #'lr_param': {"lr": args.lr,
                                        #'gamma': 0.001, # 이거 0.01이여야 하지 않나?
                                        #'power': 0.75
                                        #}
    
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])

    schedule_param = optimizer_config['lr_param']

    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]] # return optimizer


    ####################################################
    # Train
    ####################################################

    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    transfer_loss_value = 0.0
    classifier_loss_value = 0.0
    total_loss_value = 0.0

    best_acc = 0.0

    batch_size = config["data"]["source"]["batch_size"]


    for i in range(config["num_iterations"]): # num_iterations수의 batch가 학습에 사용됨
        sys.stdout.write("Iteration : {} \r".format(i))
        sys.stdout.flush()

        loss_params = config["loss"]

        base_network.train(True)
        ad_net.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target = inputs_target.cuda()

        inputs = torch.cat((inputs_source, inputs_target), dim = 0)

        features, outputs, tau, cur_mean_source, cur_mean_target, output_mean_source, output_mean_target = base_network(inputs)

        softmax_out = nn.Softmax(dim=1)(outputs)

        outputs_source = outputs[:batch_size]
        outputs_target = outputs[batch_size:]

        if config['method'] == 'CDAN+E' or config['method'] == 'CDAN_TransNorm':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method'] == 'DANN':
            pass # 나중에 정리하기
        else:
            raise ValueError('Method cannot be recognized')

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss

        total_loss.backward()
        optimizer.step()

        #tensor_writer.add_scalar('total_loss', total_loss.i )
        #tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        #tensor_writer.add_scalar('transfer_loss', transfer_loss, i)


        ####################################################
        # Test
        ####################################################
        if i % config["test_interval"] == config["test_interval"] - 1:
            # test interval 마다
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, base_network)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
                ACC = round(best_acc, 2) * 100
                torch.save(best_model, os.path.join(config["output_path"], "iter_{}_model.pth.tar".format(ACC) ))
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)


        ####################################################
        # Model Save
        ####################################################
        #if i % config["snapshot_interval"] == 0:
            #torch.save(nn.Sequential(base_network), os.path.join(config["output_path"],"iter_{:05d}_model.pth.tar".format(i)))



def image_classification_test(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            sys.stdout.write("Test : {} \r".format(i))
            sys.stdout.flush()

            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            labels = data[1]

            _, outputs = model(inputs, mode = 'test')

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()),0)
                all_label = torch.cat((all_label, labels.float()),0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


if __name__ == "__main__":
    train(config)





