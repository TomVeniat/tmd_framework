from src.experiments.IgniteExperiment import IgniteExperiment
from .classify import MNISTExperiment


def get_experiment_by_name(name):
	if name == 'mnist':
		return MNISTExperiment
	elif name == 'mnist_ignite':
		return IgniteExperiment

	raise NotImplementedError(name)

def init_experiment(_name, **kwargs):
	return get_experiment_by_name(_name)(**kwargs)