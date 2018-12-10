"""from https://github.com/pytorch/ignite"""
import sys

import torch
from torch._six import string_classes

IS_PYTHON2 = sys.version_info[0] < 3

if IS_PYTHON2:
    import collections
else:
    import collections.abc as collections


def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def size(input_, dim=0):
    if isinstance(input_, torch.Tensor):
        return input_.shape[dim]
    elif isinstance(input_, collections.Mapping):
        return input_[next(iter(input_))].shape[dim]
    elif isinstance(input_, collections.Sequence):
        return input_[0].shape[dim]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))


def convert_tensor(input_, device=None, non_blocking=False):
    """Move tensors to relevant device."""
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking) if device else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors.
    """
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(input_, input_type, func):
    """Apply a function on a object of `input_type` or mapping, or sequence of objects of `input_type`.
    """
    if isinstance(input_, input_type):
        return func(input_)
    elif isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_type(sample, input_type, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [apply_to_type(sample, input_type, func) for sample in input_]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))


def to_onehot(indices, num_classes):
    """Convert a tensor of indices to a tensor of one-hot indicators."""
    onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)
