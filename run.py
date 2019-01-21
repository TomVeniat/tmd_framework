import os

from src.utils.run import sacred_run
from src.experiments import init_experiment
from src.datasets import init_dataset
from src.modules import init_module
from src.optimizers import init_optimizer
from src.utils import external_resources as external, get_exp_name, VISDOM_CONF_PATH, MONGO_CONF_PATH, LOCAL_SAVE_PATH
from sacred.observers import FileStorageObserver


# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()
from src.utils.sacred.LogObserver import LogObserver

def init_and_run(experiment, modules, datasets, optimizers, _run):
    # _run.config['visdom_conf'] =

    # initializing datasets
    dsets = {}
    for dataset_name, dataset_config in datasets.items():
        dsets[dataset_name] = init_dataset(**dataset_config)

    # initializing modules
    mods = {}
    for module_name, module_config in modules.items():
        module_config['obs_size'] = module_config.get('obs_size', dsets['gym_env'].observation_space.shape[0])
        module_config['action_size'] = module_config.get('action_size', dsets['gym_env'].action_space.n)
        mods[module_name] = init_module(**module_config)

    # initializing optimizers
    optims = {}
    for optimizer_name, optimizer_config in optimizers.items():
        optims[optimizer_name] = init_optimizer(mods, **optimizer_config)

    # initializing experiment and running it
    return init_experiment(**mods, **dsets, **optims, **experiment, _run=_run).run()


def experiment_hook(experiment):
    if os.path.isfile(VISDOM_CONF_PATH):
        visdom_conf = external.load_conf(VISDOM_CONF_PATH)
        experiment.observers.append(LogObserver.create(visdom_conf))

    if os.path.isfile(MONGO_CONF_PATH):
        experiment.observers.append(external.get_mongo_obs(mongo_path=MONGO_CONF_PATH))
    else:
        experiment.observers.append(FileStorageObserver.create(LOCAL_SAVE_PATH))


if __name__ == '__main__':
    sacred_run(init_and_run, name='Dynamic Agent', ex_hook=experiment_hook)
