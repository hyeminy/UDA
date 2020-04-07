
import os
import pickle
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler

from options import args


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.verbose:
        print('Architecture: {}'.format(args.arch))

    if args.arch == 'vgg16':
        model = models.__dict__[args.arch](sobel=args.sobel)
        fd = int(model.top_layer.weight.size()[1])
        model.top_layer = None
        #print(model)
        #print(model.features)
        model.features = torch.nn.DataParallel(model.features) # for multi gpu
        model.cuda()
        cudnn.benchmark = True

    else:
        print('------------Resnet50-----------------')
        model = models.__dict__[args.arch]()
        print(model)
        fd = int(model.fc.weight.size()[1])
        #print(fd)
        model.fc = None
        #print(model)
        #print(model.features)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        filter(lambda x : x.requires_grad, model.parameters()),
        lr = args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd
    )
    # filter은 내부 buitins.py에 정의된 함수

    criterion = nn.CrossEntropyLoss().cuda()

    ''' optionally resume from a checkpoint
    if args.resume: # resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume) # args.resume :  path to checkpoint (default: None)
            args.start_epoch = checkpoint['epoch']
            # remove top_layer parameters from checkpoint
            for key in checkpoint['state_dict']:
                if 'top_layer' in key:
                    del checkpoint['state_dict'][key]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    '''

    # creating checkpoint repo
    exp_check = os.path.join(args.exp_output, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    clustering_log = Logger(os.path.join(args.exp_output, 'clusters')) #Logger class 생성

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    if args.verbose: # 이게 무슨 뜻인지 모르겠다
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)
    # Kmeans clustering 생성

    for epoch in range(args.start_epoch, 1):
    #for epoch in range(args.start_epoch, args.epochs): 원래 코드
        end = time.time()

        if args.arch == 'vgg16':
            model.top_layer = None
            print(model.classifier)
            print('-----------------------------------------------------------------')
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1]) #relu를 제거
            print(model.classifier)

        else: # resnet50 의 경우
            model.fc = None


        # 전체 데이터셋의 feauture를 계산산
        features = compute_features(dataloader, model, len(dataset))

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)











def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    # 아래 부분 기존의 코드에서 최신 pytorch 버전으로 수정함
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            #input_var = torch.autograd.Variable(input_tensor.cuda(), volatile=True), 예전 파이토치 버전
            input_var =input_tensor.cuda()

            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32') # 모든 feature를 담을 것 생성 (N,4096)

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and (i % 200) == 0:
                print('{0} / {1}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                      .format(i, len(dataloader), batch_time=batch_time))
    return features

if __name__ == '__main__':
    main(args)

