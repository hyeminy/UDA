from __future__ import print_function
import argparse
import torch


parser = argparse.ArgumentParser(description='SWD')

parser.add_argument('--proj_dim', type = int, default = 128, help='The number of radial projection')

parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--test_batch_size', type = int, default= 32)

parser.add_argument('--epochs', type = int, default=50)

parser.add_argument('--lr', type = float, default=0.001)
parser.add_argument('--momentum', type = float, default=0.9)
parser.add_argument('--optimizer', type = str, default='momentum' ,choices=['momentum', 'adam', 'Adadelta'])

parser.add_argument('--cuda', action = 'store_true', default=True) # 이 argument 잘 모르겠다

parser.add_argument('--seed', type = int, default=1)

parser.add_argument('--log_interval', type = int, default=50) # 50 batches 마다 store

parser.add_argument('--num_k', type = int, default= 4, help='how many steps to repeat the generator update')

parser.add_argument('--num_layer', type = int, default=2, help = 'how many layers for classifier')

parser.add_argument('--name', type = str, default='board', help = 'board dir')

parser.add_argument('--save_path', type = str, default='save/mcd')

parser.add_argument('--source_path', type = str, default="", help = 'directory pf source datasets')
parser.add_argument('--target_path', type = str, default='', help='directory of source datasets')

parser.add_argument('--resnet', type=str, default='50', choices=['18', '50', '101', '152'])

parser.add_argument('--num_classes', type = int, default=13)
parser.add_argument('--num_unit', type = int, default=2048)
parser.add_argument('--prob', type=float, default=0.5, help = 'dropout probability')
parser.add_argument('--middle', type = int , help = 'fc layer hyperparameter')

args = parser.parse_args()

config = {}

config['proj_dim'] = args.proj_dim
config['cuda'] = args.cuda and torch.cuda.is_available()
config['source_path'] = args.source_path
config['target_path'] = args.target_path
config['num_k'] = args.num_k # generate repeat 수
config['num_layer'] = args.num_layer
config['batch_size'] = args.batch_size
config['save_path'] = args.save_path + '_' + str(args.num_k)
config['seed'] = args.seed
config['resnet'] = args.resnet
config['args.lr'] = args.lr
config['optimizer'] = args.optimizer
config['num_epoch'] = args.epochs+1

config['num_classes'] =args.num_classes
config['num_unit'] = args.num_unit
config['prob'] = args.prob
config['middle'] = args.middle
