import numpy as np

from datasets.base import CombinedDataSEt
from datasets.visda import VisdaSource, VisdaTarget


DATA_SETS = {
    visdasource : VisdaSource,
    visdatarget : VisdaTarget

}
# Office31 dataset에 대해서도 추가하기

def datset_factory(dataset_code, transform_type, is_train = True, **kwargs):
    cls = DATA_SETS[dataset_code]
    if is_train:
        transform = cls.train_transform_config(transform_type)

    else:
        transform = cls.eval_transform_config(transform_type)



def dataloaders_factory(args):
    source_train_dataset = dataset_factory(args.source_dataset_code,
                                           args.transform+type+'_source',
                                           is_train = True)
