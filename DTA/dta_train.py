from dta_options import args
from trainser.dta_trainer import DTATrainer

from misc import set_up_gpu, fix_random_seed_as, create_experiment_export_folder


def main(args, trainer_cls):

    export_root, args = _setup_experiments(args)

    dataloaders = dataloaders_factory(args)









def _setup_experiments(args):
    set_up_gpu(args)
    fix_random_seed_as(args.random_seed)
    export_root = create_experiment_export_folder(args)






if __name__ == "__main__":
    main(args, DTATrainer)