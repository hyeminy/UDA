import torch

from datasets import DATA_SETS
from models.visda_architectures import create_resnet_lower, create_resnet_upper

def create_feature_extractor(args):
    encoder = create_resnet_lower(args.model) #args.model = 'resnet50'