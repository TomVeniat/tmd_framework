import torch


def size(input_, dim=0):
    """get batch size"""
    if isinstance(input_, torch.Tensor):
        return input_.shape[dim]
    elif isinstance(input_, collections.Mapping):
        return input_[next(iter(input_))].shape[dim]
    elif isinstance(input_, collections.Sequence):
        return input_[0].shape[dim]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))