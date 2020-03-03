from __future__ import print_function
from mcd_options import config

import torch
import torch.nn as nn
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
    #######################################################
    # ENV
    #######################################################
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(config['seed'])

    if config['cuda']:
        torch.cuda.manual_seed(config['seed'])

    save_path = config['save_path']

    #####################################################
    # DATA
    #####################################################
    source_path = config['source_path']
    target_path = config['target_path']

    num_k = config['num_k']

    num_layer = config['num_layer']

    batch_size = config['batch_size']

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
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ]
        ),
    }
    dsets = {
        source_path : datasets.ImageFolder(os.path.join(source_path),
                                           data_transforms[source_path]),
        target_path : datasets.ImageFolder(os.path.join(target_path),
                                           data_transforms[target_path])
    }

    train_loader = CVDataLoader()
    train_loader.initialize(dsets[source_path], dsets[target_path], batch_size) #CVDataLoader.initialize
    dataset = train_loader.load_data() #CVDataLoader.load_data()


    test_loader = CVDataLoader()
    #opt = args
    test_loader.initialize(dsets[source_path], dsets[target_path], batch_size, shuffle=True)
    dataset_test = test_loader.load_data()

    dset_sizes = {
        source_path : len(dsets[source_path]),
        target_path : len(dsets[target_path])
    }

    dset_classes = dsets[source_path].classes
    print('classes' + str(dset_classes))


    option = 'resnet' + config['resnet']
    G = ResBase(option)
    F1 = ResClassifier(num_classes = config['num_classes'], num_layer = config['num_layer'], num_unit=config['num_unit'], prob=config['prob'], middle=config['middle'])
    F2 = ResClassifier(num_classes = config['num_classes'], num_layer = config['num_layer'], num_unit=config['num_unit'], prob=config['prob'], middle=config['middle'])

    F1.apply(weights_init)
    F2.apply(weights_init)

    lr = config['lr']

    if config['cuda']:
        G.cuda()
        F1.cuda()
        F2.cuda()

    if config['optimizer'] == 'momentum':
        optimizer_g = optim.SGD(
            list(G.features.parameters()),
            lr = config['lr'],
            weight_decay=0.0005
        )

        optimizer_f = optim.SGD(
            list(F1.parameters()) + list(F2.parameters()),
            momentum=0.9,
            lr = config['lr'],
            weight_decay = 0.0005
        )

    elif config['optimizer'] == 'adam':
        optimizer_g = optim.Adam(G.features.parameters(),
                                 lr=config['lr'],
                                 weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()),
                                 lr=config['lr'],
                                 weight_decay=0.0005)

    else:
        optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)


    criterion = nn.CrossEntropyLoss().cuda()
    for ep in range(config['num_epoch']):
        G.train()
        F1.train()
        F2.train()

        for batch_idx, data in enumerate(dataset):
            if batch_idx * batch_size > 30000:
                break # 이 부분 왜 있는지 확인

            if config['cuda']:
                data1 = data['S']
                target1 = data['S_label']

                data2 = data['T']
                target2 = data['T_label']

                data1, target1 = data1.cuda(), target1.cuda()
                data2, target2 = data2.cuda(), target2.cuda()

            eta = 1.0
            data = Variable(torch.cat((data1, data2), 0 ))
            target1 = Variable(target1)

            # Step A : source data로 G, F1,F2 학습시키는 과정
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data) # source, target data 같이 입력

            output1 = F1(output)
            output_s1 = output1[:batch_size, :] # source data 부분
            loss1 = criterion(output_s1, target1)  # source data의 cross entropy 계산

            output_t1 = output1[batch_size:, :] # target data logit 부분
            output_t1 = F.softmax(output_t1) # target data softmax 통과
            entropy_loss = -torch.mean(
                                        torch.log(torch.mean(output_t1,0)+1e-6)
            )

            output2 = F2(output)
            output_s2 = output2[:batch_size, :] # source data
            loss2 = criterion(output_s2, target1) # source data의 cross entropy 계산

            output_t2 = output2[batch_size:, :] # target data logit 부분
            output_t2 = F.softmax(output_t2) # target data softmax 통과
            entropy_loss = entropy_loss - torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            # 두 F1, F2의 entropy를 더한다

            all_loss = loss1 + loss2 + 0.01*entropy_loss # 이 entropy loss가 논문에서는 class balance loss??
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B: F1, F2들의 target data에 대한 output의 차이가 max되도록 F1, F2를 트레인
            # G의 파라메터들은 고정
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data)
            output1 = F1(output)
            output_s1 = output1[:batch_size, :]
            loss1 = criterion(output_s1, target1)

            output_t1 = output1[batch_size:, :]
            output_t1 = F.softmax(output_t1)
            entropy_loss = -torch.mean(torch.log(torch.mean(output_t1,0) + 1e-6))

            output2 = F2(output)
            output_s2 = output2[:batch_size, :]
            loss2 = criterion(output_s2, target1)

            output_t2 = output2[batch_size:, :]
            output_t2 = F.softmax(output_t2)
            entropy_loss = entropy_loss - torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))

            F_loss = loss1 + loss2 - eta*loss_dis + 0.01*entropy_loss
            F_loss.backward()
            optimizer_f.step()


            # Step C : G를 train, F1, F2의 ouput의 discrepancy가 작아지도록 G를 학습
            # 이 단계를 여러번 수행한다

            for i in range(num_k):
                optimizer_g.zero_grad()

                output = G(data)
                output1 = F1(output)
                output_s1 = output1[:batch_size, :]
                loss1 = criterion(output_s1, target1)

                output_t1 = output1[batch_size:, :]
                output_t1 = F.softmax(output_t1)
                entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                # torch.mean(input, dim=0) 각 컬럼별 평균계산, 왜 mean을 계산하는 거지? 이 부분 이해가 안됨

                output2 = F2(output)
                output_s2 = output2[:batch_size, :]
                loss2 = criterion(output_s2, target1)

                output_t2 = output2[batch_size:, :]
                output_t2 = F.softmax(output_t2)
                entropy_loss = entropy_loss - torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

                loss_dis = torch.mean(torch.abs(output_t1 - output_t2)) #왜 여기서는 entropy loss를 구현하지 않았지?
                loss_dis.backward()
                optimizer_g.step()

            if batch_idx % config['log_interval'] ==0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                        ep, batch_idx * len(data), 70000,
                            100. * batch_idx / 70000, loss1.data[0], loss2.data[0], loss_dis.data[0],
                        entropy_loss.data[0]))

            if batch_idx == 1 and ep>1:
                test(test_loader,dataset_test, ep, config)



def test(test_loader,dataset_test, epoch, config):
    G.eval()
    F1.eval()
    F2.eval()

    test_loss = 0
    correct = 0
    correct2 = 0
    size = 0

    for batch_idx, data in enumerate(dataset_test):

        if batch_idx*batch_size > 5000:
            break
        data2 = data['T']
        target2 = data['T_label']

        if config['cuda']:
            data2 = data2.cuda()
            target2 = target2.cuda()

        with torch.no_grad():
            data2 = Variable(data2)
            target2 = Variable(target2)

        output = G(data2)
        output1 = F1(output)
        output2 = F2(output)

        test_loss = test_loss + F.nll_loss(output1, target2).data[0] # The negative log likelihood loss.
        pred = output1.data.max(1)[1] # max 값
        correct = correct + pred.eq(target2.data).cpu().sum()

        pred = output2.data.max(1)[1]
        correct2 = correct + pred.eq(target2.data).cpu().sum()

        k = target2.data.size()[0]

        size = size + k

    test_loss = test_loss / len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
        test_loss, correct, size,
        100. * correct / size, 100. * correct2 / size))

    value = max(100. * correct / size, 100. * correct2 / size)

    if  value>60:
        torch.save(F1.state_dict(), config['save_path']+"_"+config['resent']+str(value)+'_'+'F1.pth')
        torch.save(F2.state_dict(), config['save_path'] + "_" + config['resent'] + str(value) + '_' + 'F2.pth')
        torch.save(G.state_dict(), config['save_path'] + "_" + config['resent'] + str(value) + '_' + 'G.pth')




if __name__ == "__main__":
    train(config)
