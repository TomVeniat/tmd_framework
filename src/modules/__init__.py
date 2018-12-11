from functools import partial

from .classifiers import MnistClassifier
from .SAGAN import init_net, ResNetGenerator, ResNetDiscriminator


def get_module_by_name(name, **kwargs):
	if name == 'sagan_gen':
		return init_net(ResNetGenerator, **kwargs)
	elif name == 'sagan_dis':
		return init_net(ResNetDiscriminator, **kwargs)
	raise NotImplementedError(name)


def init_module(_name, **kwargs):
	return get_module_by_name(_name, **kwargs)
