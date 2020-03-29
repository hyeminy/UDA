import os
import numpy as np
import random
from datetime import date

import torch
import torch.backends.cudnn as cudnn




def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))
    if torch.cuda.is_available():
        args.device = 'cuda'


def fix_random_seed_as(random_seed):
    if random_seed == -1:
        random_seed = np.random.randing(100000)
        print("RANDOM SEED: {}".format(random_seed))

    random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    np.random.seed(random_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

def create_experiment_export_folder(args):
    experiment_dir = args.experiment_dir
    experiment_description = args.experiment_description

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)




def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir,
                                   (experiment_description + "_" +str(date.today())
                                    )
                                   )
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path



def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx
