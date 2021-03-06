import os
import json 
from pathlib import Path
from datetime import datetime

def load_config_file():
    conf_path = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'config.json')
    with open(conf_path , 'r') as f:
        config_content = json.load(f)
    return config_content

class ConfigManager():
    '''
    ### Configuration class for FRCN_model application. Reads config.json file. ### 
    '''
    
    config = load_config_file()
    datetime_format = "%d%m%Y_%H%M%S"

    @staticmethod
    def get_now():
        return datetime.now().strftime(ConfigManager.datetime_format)
    
    @staticmethod
    def get_base_path():
        if ConfigManager.config['EDEN']:
            return ConfigManager.config['paths']['base_path_eden']
        else:
            return ConfigManager.config['paths']['base_path']


    @staticmethod
    def get_data_path():
        if ConfigManager.config['EDEN']:
            return ConfigManager.config['paths']['spec_paths']['data_path_eden']
        else:
            return os.path.join(ConfigManager.get_base_path(), ConfigManager.config['paths']['spec_paths']['data_path'])


    @staticmethod
    def get_tuning_results_path():
        return os.path.join(ConfigManager.get_base_path(), ConfigManager.config['paths']['spec_paths']['tuning_results_path'])


    @staticmethod
    def get_dataset_path(dataset):
        return os.path.join(ConfigManager.get_data_path(), ConfigManager.config['paths']['spec_paths']['datasets'][dataset])


    @staticmethod
    def get_checkpoints_path():
        # if ConfigManager.config['EDEN']:
        #     return ConfigManager.config['paths']['spec_paths']['checkpoints_path_eden']
        return os.path.join(ConfigManager.get_base_path(), ConfigManager.config['paths']['spec_paths']['checkpoints_path'])


    @staticmethod
    def get_models_path():
         return os.path.join(ConfigManager.get_base_path(), ConfigManager.config['paths']['spec_paths']['models_path'])

    @staticmethod
    def get_prelabels_path():
        return os.path.join(ConfigManager.get_data_path(), ConfigManager.config['paths']['spec_paths']['prelabels'])


    @staticmethod
    def is_eden():
        return ConfigManager.config['EDEN']


    @staticmethod
    def get_DATA_HUB():
        pass
