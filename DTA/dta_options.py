import argparse
import json

#from datasets import DATA_SETS

parser = argparse.ArgumentParser(description='Options for DTA')

#########################
# Load Template
#########################
parser.add_argument('--config_path', type=str, default='', help='config json path')

#########################
# General Train Settings
#########################

parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epoch', type = int, default = 20)
parser.add_argument('--num_gpu', type = int, default =1 )
parser.add_argument('--device_idx', type = str, default = '0, 1, 2, 3')
parser.add_argument('--device', type = str, default= 'cuda')
parser.add_argument('--weight_decay', type = float, default = 0.0005 )
parser.add_argument('--decay_step', type = int, default = 10)
parser.add_argument('--momentum', type = float, default = 0.9)
parser.add_argument('--gamma', type = float, default = 0.5)
parser.add_argument('--log_period_as_iter', type = int, default = 10000)
parser.add_argument('--validation_period_as_iter', type = int, default= 30000 )
parser.add_argument('--test', type = bool, default=False) # Test인지 Train인지
parser.add_argument('--batch_size', type = int, default= 128)
parser.add_argument('--source_dataset_code', type = str, default='visda_source', choices=['visda_source', 'amazon', 'dslr', 'webcam'])
parser.add_argument('--target_dataset_code', type = str, default='visda_target', choices=['visda_target', 'amazon', 'dslr', 'webcam'])
parser.add_argument('--source_path', type = str, default='')
parser.add_argument('--target_path', type = str, default='')
parser.add_argument('--tranfrom_type_source', type = str, default='visda_standard_source')
parser.add_argument('--transform_type_target',type = str, default='visda_standard_target')
parser.add_argument('--transform_type_test',type = str, default='visda_standard')
parser.add_argument('--classifier_ckpt_path', type = str, default="")
parser.add_argument('--encoder_ckpt_path', type = str, default="")
parser.add_argument('--pretrain', type = str, default="")
parser.add_argument('--optimizer', type = str, default='SGD')
parser.add_argument('--model', type = str, default="resnet50")
parser.add_argument('--rampup_length', type = int, default=20)
parser.add_argument('--source_rampup_length', type = int, default=1)
parser.add_argument('--random_seed', type = int, default=12345)
parser.add_argument('--target_consistency_loss', type = str, default="kld")
parser.add_argument('--source_consistency_loss', type = str, default="l2")
parser.add_argument('--train_mode', type = str, default="dta")

#########################
# Adversarial Dropout Settings
#########################

parser.add_argument('--target_consistency_weight', type = float, default=2)
parser.add_argument('--source_consistency_weight', type = float, default=1)
parser.add_argument('--target_fc_consistency_weight', type = float, default=2)
parser.add_argument('--source_fc_consistency_weight', type = float, default=1)
parser.add_argument('--target_cnn_consistency_weight', type = float, default=2)
parser.add_argument('--source_cnn_consistency_weight', type=float, default=1)
parser.add_argument('--cls_balance_weight', type = float, default=0.01)

parser.add_argument('--entmin_weight', type=float, default=0.02)
parser.add_argument('--delta', type=float, default=0.01)
parser.add_argument('--cnn_delta', type=float, default=0.01)
parser.add_argument('--fc_delta', type=float, default=0.1)
parser.add_argument('--source_delta', type=float, default=0.0025)
parser.add_argument('--source_delta_fc', type=float, default=0.1)

#########################
# VAT settings
#########################

parser.add_argument('--use_vat', type = bool, default=True)
parser.add_argument('--xi', type= float, default=1e-06)
parser.add_argument('--ip', type=int, default=1)
parser.add_argument('--eps', type=float, default=15)
parser.add_argument('--source_vat_loss_wegiht', type=float, default=0.0)
parser.add_argument('--target_vat_loss_weight', type=float, default=0.2)

#########################
# Experiment Logging Settings
#########################

parser.add_argument('--experiment_dir', type=str, default="resnet50_experiments")
parser.add_argument('--experiment_description', type=str, default="res50_dta_vat")
parser.add_argument('--checkpoint_period', type=int, default=1)

#########################
# office 31 Data path
#########################
parser.add_argument('--amazon_path', type=str, default='')
parser.add_argument('--dslr_path', type=str, default='')
parser.add_argument('--webcam_path', type=str, default='')

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