from __future__ import print_function, absolute_import

from mmt_options import config

import random
import numpy as np
import sys
import os.path

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


from mmt import datasets
from mmt import models
from mmt.trainers import MMTTrainer
from mmt.evaluators import Evaluator, extract_features
from mmt.utils.data import IterLoader
from mmt.utils.data import transforms as T
from mmt.utils.data.sampler import RandomMultipleGallerySampler
from mmt.utils.data.preprocessor import PreProcessor
from mmt.utils.logging import Logger
from mmt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict



def main(config):

    if config['seed'] is not None:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        cudnn.deterministic = True

    main_worker(config)

def main_worker(confg):
    global start_epoch, best_mAP

    cudnn.benchmark = True #

    sys.stdout = Logger(os.path.join(config['logs_dir'], 'log.txt'))
    print("==========\nArgs:{}\n==========".format(config))

    #######################################################
    # create test data loaders
    #######################################################

    if config['iters'] > 0:
        iters = config['iters']
    else:
        iters = None

    #dataset_target = get_data(config['dataset_target'], config['data_dir'])
    test_loader_target = get_test_loader(
                                            config['data_dir'],
                                            config['resize_size'],
                                            config['batch_size'],
                                            config['workers'])

    #######################################################
    # create model
    #######################################################

    model_1, model_2, model_1_ema, model_2_ema = create_model(config)







def get_test_loader(dataset, resize_size, batch_size, workers, testset=None):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
                                T.resize(resize_size, resize_size),
                                T.ToTensor(),
                                normalize
                                ])

    test_loader = DataLoader(
                                PreProcessor
    )

    images = datasets.ImageFolder(dataset, test_transformer)
    test_loader = DataLoader(
                                images,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True
                            )

    return test_loader


def create_model(config):
    model_1 = models.create(
                                config['model'],

    )







if __name__ == "__main__":
    main(config)