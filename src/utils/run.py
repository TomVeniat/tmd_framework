from os.path import join
from copy import deepcopy
from sacred import Experiment
from sacred.utils import recursive_update
from sacred.config import load_config_file


# functions for generalizing ingredients

# def nested_update(d, sub_d, *keys):
#     if keys and d:
#         element = keys[0]
#         if element:
#             value = d.get(element)
#             if len(keys) == 1:
#                 recursive_update(value, sub_d)
#             else:
#                 return get_nested(value, sub_d, *keys[1:])

# def get_nested(d, *keys):
#     if keys and d:
#         element  = keys[0]
#         if element:
#             value = d.get(element)
#             if len(keys) == 1:
#                 return value
#             else:
#                 return get_nested(value, *keys[1:])

# sort of generic version of the config hook ... maybe for after...
# def default_config(config, command_name, logger):
#     default_config = {}
#     for ingredient in ['modules', 'datasets', 'experiment']:
#         fn = join(default_configs_root, ingredient + '.yaml')
#         default_ingredient_args = load_config_file(fn)
#         ingredient_config = config[ingredient]
#         # if there is only one component per-ingredient,
#         # like in experiment config
#         if 'name' in ingredient_config:
#             ingredient_config = {ingredient: ingredient_config}
#         default_config[ingredient] = {}
#         for component, component_config in ingredient_config.items():
#             default_config[ingredient][component] = {}
#             name = component_config['name']
#             if 'name' in config[ingredient]:
#                 default_config[ingredient]['args'] = default_ingredient_args[name]
#             else:
#                 default_config[ingredient][component]['args'] = default_ingredient_args[name]
#     return default_config

def get_component_configs(config, component_name, default_configs_file):
    """

    :param config: The global config given by the user.
    :param component_name: The key of the root-level element to process in the config.
    :param default_configs_file: The path of the file containing the default configs for the current component.
    :return: A dict containing the default configurations for each element under the given component.
    """
    component_configs = {}
    default_configs = load_config_file(default_configs_file)

    for name, specified_config in config[component_name].items():
            selected_config = specified_config['_name']
            component_configs[name] = default_configs[selected_config]
    return component_configs


def sacred_run(command, name='train', default_configs_root='default_configs', ex_hook=None):

    ex = Experiment(name)

    def default_config(config, command_name, logger):
        default_config = {}

        components = ['datasets', 'modules', 'optimizers']

        for comp in components:
            file_name = '{}.yaml'.format(comp)
            default_file_path = join(default_configs_root, file_name)
            default_config[comp] = get_component_configs(config, comp, default_file_path)

        # loading experiment default configs
        fn = join(default_configs_root, 'experiment.yaml')
        default_experiment_args = load_config_file(fn)
        default_args = default_experiment_args[config['experiment']['_name']]
        default_config['experiment'] = default_args

        return default_config

    if ex_hook:
        ex_hook(ex)

    ex.config_hook(default_config)
    ex.main(command)
    ex.run_commandline()