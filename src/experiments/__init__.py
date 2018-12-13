from .unpaired import UnpairedExperiment

def get_experiment_by_name(name):
    if name == 'unpaired':
        return UnpairedExperiment
    raise NotImplementedError(name)


def init_experiment(_name, **kwargs):
    return get_experiment_by_name(_name)(**kwargs)