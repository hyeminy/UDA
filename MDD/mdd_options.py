import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument('--init_lr', type = float, default=0.004)
parser.add_argument('--optim_type', type=str, default='sgd')
parser.add_argument('--lr', type = float, default=0.004)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--nesterov', type=bool, default=True)
parser.add_argument('--lr_type', type=str, default='inv')
parser.add_argument('--gamma', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=0.75)

parser.add_argument('--dataset', default='Office-31', type=str)
parser.add_argument('--source_path', default=None, type=str)
parser.add_argument('--target_path', default=None, type=str)

parser.add_argument('--class_num', default=31, type=int)
parser.add_argument('--width', default=1024, type=int)
parser.add_argument('--srcweight', default=4, type=int)
parser.add_argument('--is_cen', default=False, type=bool)

parser.add_argument('--base_net', default='resnet50', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--resize_size', default=256, type=int)
parser.add_argument('--crop_size', default=224, type=int)
parser.add_argument('--max_iter', default=100000, type=int)
parser.add_argument('--eval_iter', default=1000, type=int)
parser.add_argument('--output_path', default=None, type=str)

args = parser.parse_args()

config={}
config['init_lr'] =args.init_lr
config['optim'] = {'type': args.optim_type,
                   'params': {'lr': args.lr,
                              'momentum': args.momentum,
                              'weight_decay': args.weight_decay,
                              'nesterov': args.nesterov}
                   }
config['lr_scheduler'] = {
                            'type': args.lr_type,
                            'gamma' : args.gamma,
                            'decay_rate' : args.decay_rate
                          }
config['dataset'] = args.dataset
config['source_path'] = args.source_path
config['target_path'] = args.target_path

config['class_num'] = args.class_num
config['width'] = args.width
config['srcweight'] = args.srcweight
#config['is_cen'] = args.is_cen
config['is_cen'] = False

config['base_net'] = args.base_net

config['batch_size'] = args.batch_size
config['resize_size'] = args.resize_size
config['crop_size'] = args.crop_size
config['max_iter'] = args.max_iter
config['eval_iter'] = args.eval_iter
config['output_path'] = args.output_path

if not os.path.exists(config["output_path"]):
	os.mkdir(config["output_path"])




