import torch.utils.data as data
from ..modules.corruption import init_corruption


def corrupted(dataset, *args, **kwargs):
    corruption = init_corruption(kwargs['corrpution'])
    return CorruptedDataset(dataset, corruption)

class CorruptedDataset(data.dataset):

    def __init__(self, dataset, corruption):
        self.dataset = dataset
        self.corruption = corruption
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):

        if len(self.dataset[item]) == 1:
            return {'sample'}
