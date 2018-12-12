import torch.optim as optim


def get_optim_by_name(name):
    if name == 'sgd':
        return optim.SGD
    elif name == 'adam':
        return optim.Adam
    raise NotImplementedError(name)


def get_lr_scheduler_by_name(name):
    if name == 'None':
        return None
    elif name == 'exponential':
        return optim.lr_scheduler.exponential
    else:
        raise NotImplementedError(name)


def init_lr_scheduler(optimizer, _name, **kwargs):
    return get_lr_scheduler_by_name(_name)(optimizer, **kwargs)


def init_optimizer(modules, _name, _modules, lr_scheduler=None, **kwargs):
    if isinstance(_modules, str):
        _modules = [_modules]
    
    parameters = []
    for name in _modules:
        module = modules[name]
        parameters += list(module.parameters())
    optim = get_optim_by_name(_name)(parameters, **kwargs)
    
    return optim