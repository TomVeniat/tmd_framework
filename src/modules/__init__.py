from .classifiers import MnistClassifier, Policy


def get_module_by_name(name):
    if name == 'mnist_classifier':
        return MnistClassifier
    elif name == 'policy':
        return Policy
    raise NotImplementedError(name)


def init_module(_name, **kwargs):
    return get_module_by_name(_name)(**kwargs)
