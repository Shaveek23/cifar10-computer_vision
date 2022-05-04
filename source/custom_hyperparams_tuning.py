from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import ParameterGrid
import copy

'''
config = {
    'net': [ # list of models to be tested: {type: class_name, arguments: dict_of_class_init_arguments}
        {
            'type': VGGLike,
            'arguments': {
               'p_last_droput': [0.2, 0.3, 0.4]
            }
        },
        {
            'type': VGGLike2,
            'arguments': {
               'p_last_droput': [0.1, 0.5, 0.6]
            }
        }
    ],
    'optimizer': [ # list of optimizers to be tested: {type: class_name, arguments: dict_of_class_init_arguments}
        {
            'type': torch.optim.Adam,
            'arguments': {
                'lr': [0.01, 0.001]
            }
        }
    ],
    'transform': [
        {
            'type': {
                'train': AudioTrainTrasformersFactory.get_transformer_spectogram_aug,
                'test': AudioTestTrasformersFactory.get_transformer_spectogram_aug
            },
            'arguments': {
                'new_freq': [8_000, 16_000] 
            }
        },
        {
            'type': {
                'train': TrainTrasformersFactory.get_transformer_spectogram,
                'test': TestTransformersFactory.get_transformer_spectogram
            },
            'arguments': {
                'spect_size': [32, 64] 
            }
        }
    ],
    'batch_size': [32, 64, 128]
}
'''

def flatten_config_dict(config):
    config_copy = copy.deepcopy(config)
    grid_dict = {}
    for k, v in config_copy.items():
        
        if isinstance(v[0], dict):
            concatenated = []
            for d in v:
                d['arguments'].update({'type': [d['type']]})
                concatenated += [params for params in list(ParameterGrid(d['arguments']))]
            grid_dict.update({k: concatenated})
        else:
            grid_dict.update({k: v})
    return grid_dict 


def build_transformers(transform_dict):
    transform_dict_cp = copy.deepcopy(transform_dict)
    transformers_type = transform_dict_cp.pop('type')
    transformers_kwargs = transform_dict_cp
    return transformers_type['train'], transformers_type['test'], transformers_kwargs


def build_net(net_dict):
    net_dict_cp = copy.deepcopy(net_dict)
    net_type = net_dict_cp.pop('type')
    net_kwargs = net_dict_cp
    return net_type, net_kwargs


def build_optimizer(opt_dict):
    opt_dict_cp = copy.deepcopy(opt_dict)
    opt_type = opt_dict_cp.pop('type')
    opt_kwargs = opt_dict_cp
    return opt_type, opt_kwargs


def get_trial_params(trial_dict):
    train_transform_type, test_transform_type, transform_kwargs = build_transformers(trial_dict['transform'])
    model_type, model_kwargs = build_net(trial_dict['net'])
    opt_type, opt_kwargs = build_optimizer(trial_dict['optimizer'])
    batch_size = trial_dict['batch_size']

    train_transform = train_transform_type(**transform_kwargs)
    test_transform = test_transform_type(**transform_kwargs)
    model = model_type(**model_kwargs)
    optimizer = opt_type(model.parameters(), **opt_kwargs)

    return {
        'train_transform': train_transform,
        'test_transform': test_transform,
        'model': model,
        'optimizer': optimizer,
        'batch_size': batch_size
    }


def get_trials(config, n_trials=10):
        grid_dict = flatten_config_dict(config)
        return [{'trial_objects': get_trial_params(trial_dict), 'trial_dict': trial_dict} for trial_dict in list(ParameterSampler(grid_dict, n_iter=n_trials))]
    