from .run import sacred_run

VISDOM_CONF_PATH = 'resources/visdom.json'
MONGO_CONF_PATH = 'resources/mongo.json'
LOCAL_SAVE_PATH = '/local/veniat/runs'


def get_exp_name(config, _id):
    datasets = config['datasets']
    assert len(datasets) == 1
    ds_name = list(datasets.values())[0]['_name']

    modules = config['modules']
    assert len(modules) == 1
    model_name = list(modules.values())[0]['_name']

    exp_type = config['experiment']['_name']

    return f'{exp_type}_{ds_name}_{model_name}_{_id}'
