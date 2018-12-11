import torch.utils.data as data
from ..modules.corruption import init_corruption


def corrupted(dataset, corruption):
    name = corruption.pop('name')
    corruption = init_corruption(name, **corruption)
    return CorruptedDataset(dataset=dataset, corruption=corruption)


class CorruptedDataset(data.dataset.Dataset):

    def __init__(self, dataset, corruption):
        self.dataset = dataset
        self.corruption = corruption
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        x_measurement, target = batch
        meas = self.corruption.measure(x_measurement.unsqueeze(0), seed=item)
        dict_var = {
            'sample': x_measurement,
            'target': target
        }
        dict_var["measured_sample"] = meas["measured_sample"][0]  # because the dataloader add a dimension
        return dict_var






