import os
import numpy as np

from source.utils.config_manager import ConfigManager
from source.dataloaders.onevsall_dataloadersfactory import OneVsAllDataLoadersFactory
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory

from source.training import fit



dataset_name = 'speech_recognition'
dataset_path = os.path.join(ConfigManager().get_dataset_path(dataset_name))

# Monolithic approach (one model)
def train_one_vs_one(n_classes, model, optimizer, criterion, train_transformer, test_transform, batch_size, n_epochs, device,
 is_logging=True, epoch_logging=1, is_balanced=False):

    if n_classes == 10: # only 10 known
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=False)
    
    elif n_classes == 11: # 10 known + unknown
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=True, is_balanced=is_balanced)
    
    elif n_classes == 12: # 10 known + silence + unknown 
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=True, with_unknown=True, is_balanced=is_balanced)
   
    elif n_classes == 31: # all possible classes (including silence) distinguished
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=True, with_unknown=True)
    
    elif n_classes == 30:# all possible classes (without silence) distinguished
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=True)
    
    else:
        print(f"Wrong number of classes: {n_classes}.")
        return


    train_loader = loaders_factory.get_train_loader(batch_size)
    valid_loader = loaders_factory.get_valid_loader(batch_size)
    fit(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, is_logging=is_logging, epoch_logging=epoch_logging)


def predict_one_vs_one(model, test_transform, batch_size, device, one_vs_rest_path=None):
    ''' Performs predicting of multiclassification problem, when 'one_vs_rest_path' specified it predicts only 'rest' files (fron the file)'''

    
    svr_factory = OneVsAllDataLoadersFactory(dataset_path, None, test_transform, from_file_path=one_vs_rest_path)

    test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model.predict(test_loader, device)
    filenames = test_loader.dataset._walker

    return predictions, filenames


# SILENCE VS REST
def train_silence_vs_rest(model_binary_classifier, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device):
    ''' Binary classification between silence (basic + extra): 0 and rest (known and unknown): 1'''
  
    svr_factory = OneVsAllDataLoadersFactory(dataset_path, train_transform, test_transform, one='silence')

    train_loader = svr_factory.get_train_loader(batch_size)
    valid_loader = svr_factory.get_valid_loader(batch_size)

    fit(model_binary_classifier, train_loader, valid_loader, optimizer, criterion, n_epochs, device, is_logging=True, epoch_logging=1)


def predict_silence_vs_rest(model_binary_classifier, test_transform, batch_size, device) -> str:
  

    svr_factory = OneVsAllDataLoadersFactory(dataset_path, None, test_transform)

    test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model_binary_classifier.predict(test_loader, device)

    filenames = test_loader.dataset._walker

    return predictions, filenames


# UNKNOWN VS KNOWN
def train_unknown_vs_known(model_binary_classifier, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device):
    ''' Binary classification between known and unknown (no silence), 0 - unknown, 1 - known'''
    
    svr_factory = OneVsAllDataLoadersFactory(dataset_path, train_transform, test_transform, one='unknown')

    train_loader = svr_factory.get_train_loader(batch_size)
    valid_loader = svr_factory.get_valid_loader(batch_size)

    fit(model_binary_classifier, train_loader, valid_loader, optimizer, criterion, n_epochs, device, is_logging=True, epoch_logging=1)


def predict_known_vs_unknown(model_binary_classifier, test_transform, batch_size, device, silence_vs_rest_path) -> str:
  

    svr_factory = OneVsAllDataLoadersFactory(dataset_path, None, test_transform, from_file_path=silence_vs_rest_path)

    test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model_binary_classifier.predict(test_loader, device)

    
    filenames = test_loader.dataset._walker

    return predictions, filenames



def predict_all(final_model, unknown_model, silence_model, test_transform, batch_size, device):
    
    mapping = {0:'yes', 1:'no', 2:'up', 3:'down', 4:'left', 5:'right',
                6:'on', 7:'off', 8:'stop', 9:'go' }
    next_label = 10


    if unknown_model is None:
        mapping.update({next_label:'unknown'})
        next_label += 1
    
    if silence_model is None:
        mapping.update({next_label:'silence'})

    filepath_rest = None

    if not(silence_model is None and unknown_model is None):
        abs_path = ConfigManager().get_prelabels_path()
        filename_rest = f'{ConfigManager().get_now()}_onevsall.txt'
        filepath_rest = os.path.join(abs_path, filename_rest)
    

    results = []

    if silence_model is not None:
        predicts, filenames = predict_silence_vs_rest(silence_model, test_transform, batch_size, device)
        
        silence_filepaths = np.array(filenames)[np.where(np.array(predicts) == 0)[0]]
        silence_filenames = [os.path.split(s)[-1] for s in silence_filepaths]
        labels = ['silence'] * len(silence_filenames)
        results += list(zip(silence_filenames, labels))
        save_prelabeled(filepath_rest, files=silence_filepaths)


    if unknown_model is not None:
        predicts, filenames = predict_known_vs_unknown(unknown_model, test_transform, batch_size, device, filepath_rest)
        unknown_filepaths = np.array(filenames)[np.where(np.array(predicts) == 0)[0]]
        unknown_filenames = [os.path.split(s)[-1] for s in unknown_filepaths]
        labels = ['unknown'] * len(unknown_filenames)
        results += list(zip(unknown_filenames, labels))
        save_prelabeled(filepath_rest, files=unknown_filepaths)


    predicts, filepaths = predict_one_vs_one(final_model, test_transform, batch_size, device, filepath_rest)
    filenames = [os.path.split(s)[-1] for s in filepaths]
    predicts = np.vectorize(mapping.get)(np.array(predicts))
    results += list(zip(filenames, predicts))

    if filepath_rest is not None:
        try:
            os.remove(filepath_rest)
        except OSError as e:
            print("Error: %s : %s" % (filepath_rest, e.strerror))

    return results


def save_prelabeled(filepath, files):
    with open(filepath, 'a') as f:
        for file_name in files:
            f.write(f'{file_name}\n')

    return filepath