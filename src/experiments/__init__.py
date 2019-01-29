from src.experiments.ActorCriticExperiment import ActorCriticExperiment
from src.experiments.IgniteExperiment import IgniteExperiment
from src.experiments.PolicyGradientExperiment import PolicyGradientExperiment
from .classify import MNISTExperiment


def get_experiment_by_name(name):
	if name == 'mnist':
		return MNISTExperiment
	elif name == 'mnist_ignite':
		return IgniteExperiment
	elif name == 'actor_critic':
		return ActorCriticExperiment
	elif name in ['pg', 'ac']:
		return PolicyGradientExperiment

	raise NotImplementedError(name)

def init_experiment(_name, **kwargs):
	return get_experiment_by_name(_name)(**kwargs)