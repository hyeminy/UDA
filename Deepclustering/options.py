import argparse
import os
import json


parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

#########################
# Load Template
#########################
parser.add_argument('--config_path', type=str, default='', help='config json path')

parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
#parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp_output', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')
# store_true의 의미?
# boolean: action='store_true'를 사용하여 해당하는 인자(argument)가 입력되면 True, 입력되지 않으면 False로 인식하게 됩니다.





def _load_experiments_config_from_json(args, json_path, arg_parser):
    with open(json_path, 'r') as f:
        config = json.load(f)

    for config_name, config_val in config.items():
        if config_name in args.__dict__ and getattr(args, config_name) == arg_parser.get_default(config_name):
            setattr(args, config_name, config_val)

    print("Config at '{}' has been loaded".format(json_path))

def get_parsed_args(arg_parser: argparse.ArgumentParser):
    args = arg_parser.parse_args()
    if args.config_path:
        _load_experiments_config_from_json(args, args.config_path, arg_parser)
    return args

args = get_parsed_args(parser)

args.out_file= open(os.path.join(args.exp_output, 'log.txt'), 'w') #log를 다 저장

if not os.path.exists(args.exp_output):
    os.mkdir(args.exp_output)

