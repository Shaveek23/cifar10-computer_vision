import torch
from source.project2.training_scripts import predict_all, get_confusion_matrix
import pandas as pd
import os

from source.utils.config_manager import ConfigManager
UNKNOWN_DIRS = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]

def create_kaggle_submision_file_audio(output_file,test_transform_silence,test_transform_unknown,test_transform_final, final_model, final_model_path='', unknown_model=None, unknown_model_path='', 
silence_model=None, silence_model_path='', batch_size=64, device="cpu", n_classes = None):

    final_model.load_state_dict(torch.load(
        final_model_path, map_location=torch.device(device))[0])
        
    if silence_model:
            silence_model.load_state_dict(torch.load(
        silence_model_path, map_location=torch.device(device))[0])

    if unknown_model:
            unknown_model.load_state_dict(torch.load(
        unknown_model_path, map_location=torch.device(device))[0])

    predicts = predict_all(final_model, unknown_model, silence_model,
                    test_transform_silence,test_transform_unknown,test_transform_final, batch_size, device, n_classes)
                    
    res = {'fname': 'label'}

    predict_dir = os.path.join(ConfigManager().get_base_path(), f'predictions/{output_file}/')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
   
    res.update({k: v for k, v in predicts})
    res = pd.Series(res)
    res = res.apply(lambda x : 'unknown' if x in UNKNOWN_DIRS else x)
    csv_path = os.path.join(predict_dir, f'{output_file}.csv')
    res.to_csv(csv_path, header=None)
    get_confusion_matrix(predict_dir, final_model, unknown_model, silence_model, test_transform_silence,test_transform_unknown,test_transform_final, device, n_classes)
    
