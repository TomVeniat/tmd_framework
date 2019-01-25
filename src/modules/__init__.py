from src.modules.DynamicPolicy import DynamicPolicy
from src.modules.Policy import Policy


def get_module_by_name(name):
    if name == 'classic':
        return Policy
    elif name == 'dynamic':
        return DynamicPolicy
    raise NotImplementedError(name)


def init_module(_name, **kwargs):
    return get_module_by_name(_name)(**kwargs)
