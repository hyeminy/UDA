import argparse
import os.path as osp


parser = argparse.ArgumentParser(description= "MMT Training")

###################################
# data
###################################

parser.add_argument('--dataset_target', type=str, default='webcam')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type = int, default=4)
parser.add_argument('--num_clusters', type=int, default=31)
parser.add_argument('--resize_size', type=int, default=256)


###################################
# model
###################################

parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--features', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0)

###################################
# optimizer
###################################

parser.add_argument('--lr', type=float, default=0.00035)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--alpha', type=float, default=0.999)
parser.add_argument('--moving_avg_momentum', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--soft_ce_weight', type=float, default=0.5)
parser.add_argument('--soft_tri_weight', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--iters', type=int, default=800)

###################################
# training_configs
###################################

parser.add_argument('--init_1', type=str, default='', metavar='PATH')
parser.add_argument('--init_2', type=str, default='', metavar='PATH')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--eval_step', type=int, default=1)

###################################
# path
###################################

parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--logs_dir', type=str, default='')


args = parser.parse_args()

config={}

config['dataset_target'] = args.dataset_target
config['batch_size'] = args.batch_size
config['num_workers']=args.num_workers
config['num_clusters']=args.num_clusters
config['resize_size']=args.resize_size



config['model']=args.model
config['features']=args.features
config['dropout']=args.dropout

config['lr']=args.lr
config['momentum']=args.momentum
config['alpha']=args.alpha
config['moving_avg_momentum']=args.moving_avg_momentum
config['weight_decay']=args.weight_decay
config['soft_ce_weight']=args.soft_ce_weight
config['soft_tri_weight']=args.soft_tri_weight
config['epochs']=args.epochs
config['iters']=args.iters


config['init_1']=args.init_1
config['init_2']=args.init_2
config['seed']=args.seed
config['print_freq']=args.print_freq
config['eval_step']=args.eval_step

config['data_dir']=args.data_dir
config['logs_dir']=args.logs_dir




























