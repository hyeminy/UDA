import numpy as np
from torch.utils.data import DataLoader, Subset



from datasets.base import CombinedDataSet
from datasets.visda import VisdaSource, VisdaTarget
from datasets.office31 import Amazon, Webcam, Dslr


DATA_SETS = {
    'visdasource' : VisdaSource,
    'visdatarget' : VisdaTarget,
    'amazon' : Amazon,
    'webcam' : Webcam,
    'dslr' : Dslr

}


def dataset_factory(dataset_code, transform_type, is_train = True, **kwargs):
    cls = DATA_SETS[dataset_code]
    if is_train:
        transform = cls.train_transform_config(transform_type)
    else:
        transform = cls.eval_transform_config(transform_type)

    print("{} has been created.".format(cls.code()))
    return cls(transform = transform, **kwargs) # 이 부분 transform이 어디로 가는지 모르겠다




def dataloaders_factory(args):
    source_train_dataset = dataset_factory(args.source_dataset_code,
                                           args.transform_type_source,
                                           is_train = True) #datasets 내에 있는 cls가 return이 됨
    taraget_train_dataset = dataset_factory(args.target_dataset_code,
                                            args.transform_type_target,
                                            is_train = True)

    train_dataset = CombinedDataSet(source_train_dataset, taraget_train_dataset)

    target_val_dataset = dataset_factory(args.target_dataset_code, args.transform_type_test, is_train = False)


    if args.test:
        train_dataset = Subset(train_dataset,
                               np.random.randint(0, len(train_dataset), args.batch_size * 5))

        target_val_dataset = Subset(target_val_dataset,
                                    np.random.randint(0, len(target_val_dataset), args.batch_size * 5))

    train_dataloader = DataLoader(target_val_dataset,
                                  batch_size=args.batch_size,
                                  num_workers = 16,
                                  shuffle=True,
                                  pin_memory=True)

    target_val_dataloader = DataLoader(target_val_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=16,
                                       shuffle=False,
                                       pin_memory=True)

    return {
        'train' : train_dataloader,
        'val': target_val_dataloader
        }