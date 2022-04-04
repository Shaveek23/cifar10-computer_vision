import json
import os
import torch


def get_type(object_data):
    module_name, class_name = object_data.pop('type').rsplit('.', 1)
    module_name = module_name.split('\'')[1]
    class_name = class_name.split('\'')[0]
    module = __import__(module_name, fromlist=[class_name]) 
    return getattr(module, class_name)

def get_object(object_data):
    type = get_type(object_data)
    return type(**object_data)

def get_cached_net_opt_crit(params_json_path):
    '''
        params_json_path: str - a path to the checkpoint, eg.: .\tuning_results\__train_validate_2022-03-24_18-12-52\HyperparametersTunner_DBN\checkpoint_000014
    '''
    with open(os.path.join(params_json_path.rsplit('\\', 1)[0], 'params.json'), 'r') as f:
        data = json.load(f)
    
    optimizer_data = data['net'].pop('optimizer')
    criterion_data = data['net'].pop('criterion')
    net_data = data['net']

    net = get_object(net_data)

    optimizer_data['params'] = net.parameters()
    optimizer = get_object(optimizer_data)
    
    criterion = get_object(criterion_data)

    state_file_name = data['tuning_id']


    try:
        model_state, optimizer_state, criterion_state = torch.load(os.path.join(params_json_path, state_file_name))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        criterion.load_state_dict(criterion_state)
    except ValueError:
        print(f'WARNING: No optimizer state found in the file: {params_json_path}')
        model_state, optimizer_state = torch.load(os.path.join(params_json_path, state_file_name))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    return net, optimizer, criterion