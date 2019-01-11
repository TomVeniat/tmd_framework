import gym
from torch.utils.data import DataLoader
from functools import partial

from .mnist import mnist


def get_dataset_by_name(name):
    if name == 'mnist':
        return partial(mnist, fashion=False)
    elif name == 'fashion_mnist':
        return partial(mnist, fashion=True)
    elif name == 'CartPole-v0':
        return gym.make
    raise NotImplementedError(name)


def init_dataset(_name, batch_size=None, num_workers=None, drop_last=None, shuffle=None, **kwargs):
    ds = get_dataset_by_name(_name)(**kwargs)
    if isinstance(ds, gym.Env):
        return ds
    dl = DataLoader(dataset=ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    )
    return dl