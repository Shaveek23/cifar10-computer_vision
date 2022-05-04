import torch
from source.project2.training_scripts import predict_all
import pandas as pd
UNKNOWN_DIRS = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]

def create_kaggle_submision_file_audio(output_file, test_transform, final_model, final_model_path='', unknown_model=None, unknown_model_path='', 
silence_model=None, silence_model_path='', batch_size=64, device="cpu",n_classes = None):

    final_model.load_state_dict(torch.load(
        final_model_path, map_location=torch.device(device))[0])
        
    if silence_model:
            silence_model.load_state_dict(torch.load(
        silence_model_path, map_location=torch.device(device))[0])

    if unknown_model:
            unknown_model.load_state_dict(torch.load(
        unknown_model_path, map_location=torch.device(device))[0])

    _ = predict_all(final_model, unknown_model, silence_model,
                    test_transform, batch_size, device,n_classes)
    res = {'fname': 'label'}
    
    res.update({k: v for k, v in _})
    res = pd.Series(res)
    res = res.apply(lambda x : 'unknown' if x in UNKNOWN_DIRS else x)
    res.to_csv(output_file, header=None)
