from torch.utils.data import DataLoader
from functools import partial

from .imagenet import init_imagenet
from .corrupted_dataset import corrupted


def get_dataset_by_name(name):
    elif name == 'imagenet_tiny':
        return partial(init_imagenet, tiny=True)
    elif name == 'imagenet':
        return partial(init_imagenet, tiny=False)
    raise NotImplementedError(name)


def init_dataset(_name, batch_size, num_workers, drop_last, shuffle, **kwargs):
    corruption = kwargs.pop('corruption')

    ds = get_dataset_by_name(_name)(**kwargs)

    if corruption is not None:
        ds = corrupted(ds, corruption)

    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    )
    return dl