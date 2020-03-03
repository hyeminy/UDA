import random
import torch.utils.data
from builtins import object


class CVDataLoader(object):

    def initialize(self, dataset_A, dataset_B, batch_size, shuffle=True):

        self.max_dataset_size = float("inf")

        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers=4
        )

        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers=4
        )

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False

        self.paired_data = PairedData(
            data_loader_A,
            data_loader_B,
            self.max_dataset_size,
            flip
        )


    def load_data(self):
        return self.paired_data

    def name(self):
        return 'UnalignedDataLoader'

    def __len__(self):
        return min(
            max(len(self.dataset_A), len(self.dataset_B)),
            self.opt.max_dataset_size
        )
    

class PairedData(object):
