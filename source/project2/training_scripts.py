import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import json
from os import listdir
from os.path import isfile, join
from source.custom_hyperparams_tuning import get_trials
from enum import Enum
from source.utils.config_manager import ConfigManager
from source.dataloaders.onevsall_dataloadersfactory import OneVsAllDataLoadersFactory
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory
from source.training import fit, generate_checkpoint_path
import pickle

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from tqdm import tqdm

from source.utils.confusion_matrix import save_confusion_matrix




dataset_name = 'speech_recognition'
dataset_path = os.path.join(ConfigManager().get_dataset_path(dataset_name))

# Monolithic approach (one model)


def train_one_vs_one(n_classes, model, optimizer, criterion, train_transformer, test_transform, batch_size, n_epochs, device,
                     is_logging=True, epoch_logging=1, is_balanced=True, trial_name=None, checkpoint_path=None):

    if n_classes == 10:  # only 10 known
        loaders_factory = Project2DataLoaderFactory(
            dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=False)

    elif n_classes == 11:  # 10 known + unknown
        loaders_factory = Project2DataLoaderFactory(
            dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=True, is_balanced=is_balanced)

    elif n_classes == 12:  # 10 known + silence + unknown
        loaders_factory = Project2DataLoaderFactory(
            dataset_path, train_transformer, test_transform, with_silence=True, with_unknown=True, is_balanced=is_balanced)

    # all possible classes (including silence) distinguished
    elif n_classes == 31:
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=True, with_unknown=True,
                                                    labels={'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5,
                                                            'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'zero': 10, 'one': 11, 'two': 12, 'three': 13, 'four': 14, 'five': 15, 'six': 16, 'seven': 17, 'eight': 18, 'nine': 19,
                                                            'happy': 20, 'house': 21, 'cat': 22, 'wow': 23, 'marvin': 24, 'bird': 25, 'bed': 26, 'tree': 27, 'dog': 28, 'sheila': 29, 'silence': 30})

    # all possible classes (without silence) distinguished
    elif n_classes == 30:
        loaders_factory = Project2DataLoaderFactory(dataset_path, train_transformer, test_transform, with_silence=False, with_unknown=True,
                                                    labels={'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5,
                                                            'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'zero': 10, 'one': 11, 'two': 12, 'three': 13, 'four': 14, 'five': 15, 'six': 16, 'seven': 17, 'eight': 18, 'nine': 19,
                                                            'happy': 20, 'house': 21, 'cat': 22, 'wow': 23, 'marvin': 24, 'bird': 25, 'bed': 26, 'tree': 27, 'dog': 28, 'sheila': 29})

    else:
        print(f"Wrong number of classes: {n_classes}.")
        return

    train_loader = loaders_factory.get_train_loader(batch_size)
    valid_loader = loaders_factory.get_valid_loader(batch_size)
    fit(model, train_loader, valid_loader, optimizer, criterion, n_epochs, device, is_logging=is_logging, epoch_logging=epoch_logging,
        trial_name=trial_name, checkpoint_path=checkpoint_path)


def predict_one_vs_one(model, test_transform, batch_size, device, one_vs_rest_path=None, is_valid_dataset=False):
    ''' Performs predicting of multiclassification problem, when 'one_vs_rest_path' specified it predicts only 'rest' files (fron the file)'''
    if device == 'cuda':
        model.to(device)

    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, None, test_transform, from_file_path=one_vs_rest_path)

    if is_valid_dataset:
        test_loader = svr_factory.get_valid_loader(batch_size)
    else:
        test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model.predict(test_loader, device)
    filenames = test_loader.dataset._walker

    return predictions, filenames


# SILENCE VS REST
def train_silence_vs_rest(model_binary_classifier, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device,
                          is_logging=True, epoch_logging=1, trial_name=None, checkpoint_path=None):
    ''' Binary classification between silence (basic + extra): 0 and rest (known and unknown): 1'''

    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, train_transform, test_transform, one='silence')

    train_loader = svr_factory.get_train_loader(batch_size)
    valid_loader = svr_factory.get_valid_loader(batch_size)

    fit(model_binary_classifier, train_loader, valid_loader, optimizer, criterion, n_epochs, device,
        is_logging=is_logging, epoch_logging=epoch_logging, trial_name=trial_name, checkpoint_path=checkpoint_path)


def predict_silence_vs_rest(model_binary_classifier, test_transform, batch_size, device, is_valid_dataset=False) -> str:
    if device == 'cuda':
        model_binary_classifier.to(device)

    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, None, test_transform)

    if is_valid_dataset:
        test_loader = svr_factory.get_valid_loader(batch_size)
    else:
        test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model_binary_classifier.predict(test_loader, device)

    filenames = test_loader.dataset._walker

    return predictions, filenames


# UNKNOWN VS KNOWN
def train_unknown_vs_known(model_binary_classifier, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device,
                           is_logging=True, epoch_logging=1, trial_name=None, checkpoint_path=None):
    ''' Binary classification between known and unknown (no silence), 0 - unknown, 1 - known'''

    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, train_transform, test_transform, one='unknown')

    train_loader = svr_factory.get_train_loader(batch_size)
    valid_loader = svr_factory.get_valid_loader(batch_size)

    fit(model_binary_classifier, train_loader, valid_loader, optimizer, criterion, n_epochs, device,
        is_logging=is_logging, epoch_logging=epoch_logging, trial_name=trial_name, checkpoint_path=checkpoint_path)


def predict_known_vs_unknown(model_binary_classifier, test_transform, batch_size, device, silence_vs_rest_path, is_valid_dataset=False) -> str:
    if device == 'cuda':
        model_binary_classifier.to(device)

    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, None, test_transform, from_file_path=silence_vs_rest_path)

    if is_valid_dataset:
        test_loader = svr_factory.get_valid_loader(batch_size)
    else:
        test_loader = svr_factory.get_test_loader(batch_size)

    predictions = model_binary_classifier.predict(test_loader, device)

    filenames = test_loader.dataset._walker

    return predictions, filenames




def predict_all(final_model, unknown_model, silence_model, test_transform_silence,test_transform_unknown,test_transform_final, batch_size, device, n_class=None, is_valid_dataset=False):
    
    mapping = {0:'yes', 1:'no', 2:'up', 3:'down', 4:'left', 5:'right',
                6:'on', 7:'off', 8:'stop', 9:'go' }
    next_label = 10

    if n_class == 30 or n_class == 31:
        mapping.update({
            10: 'zero',
            11: 'one',
            12: 'two',
            13: 'three',
            14: 'four',
            15: 'five',
            16: 'six',
            17: 'seven',
            18: 'eight',
            19: 'nine',
            20: 'happy',
            21: 'house',
            22: 'cat',
            23: 'wow',
            24: 'marvin',
            25: 'bird',
            26: 'bed',
            27: 'tree',
            28: 'dog',
            29: 'sheila'
        })
        next_label = 30
    elif unknown_model is None:
        mapping.update({next_label: 'unknown'})
        next_label += 1

    if silence_model is None:
        mapping.update({next_label: 'silence'})

    filepath_rest = None

    if not(silence_model is None and unknown_model is None):
        abs_path = ConfigManager().get_prelabels_path()
        filename_rest = f'{ConfigManager().get_now()}_onevsall.txt'
        filepath_rest = os.path.join(abs_path, filename_rest)

    results = []

    if silence_model is not None:


        predicts, filenames = predict_silence_vs_rest(silence_model, test_transform_silence, batch_size, device, is_valid_dataset)
        
        silence_filepaths = np.array(filenames)[np.where(np.array(predicts) == 0)[0]]

        silence_filenames = [os.path.split(s)[-1] for s in silence_filepaths]
        labels = ['silence'] * len(silence_filenames)
        results += list(zip(silence_filenames, labels))
        save_prelabeled(filepath_rest, files=silence_filepaths)

    if unknown_model is not None:

        predicts, filenames = predict_known_vs_unknown(unknown_model, test_transform_unknown, batch_size, device, filepath_rest, is_valid_dataset)
        unknown_filepaths = np.array(filenames)[np.where(np.array(predicts) == 0)[0]]
        unknown_filenames = [os.path.split(s)[-1] for s in unknown_filepaths]
        labels = ['unknown'] * len(unknown_filenames)
        results += list(zip(unknown_filenames, labels))
        save_prelabeled(filepath_rest, files=unknown_filepaths)


    predicts, filepaths = predict_one_vs_one(final_model, test_transform_final, batch_size, device, filepath_rest, is_valid_dataset)
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


def get_results_from_data(checkpoint_name):
    checkpoints_path = ConfigManager().get_checkpoints_path()
    mypath = os.path.join(checkpoints_path, checkpoint_name)
    only_dirs = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

    only_dirs = sorted(only_dirs, key=lambda x: int(x.split('_')[-1]))

    last_epoch_dir = only_dirs[-1]

    f = open(os.path.join(mypath, os.path.join(last_epoch_dir, 'results.json')))
    data = json.load(f)
    f.close()

    return data


def get_best_epoch(data, measure='Accuracy'):

    assert measure in ['Accuracy', 'Recall', 'Precision', 'F1Score'], (
        "measure should be one of: 'Accuracy', 'Recall', 'Precision', 'F1Score'")

    idx_max = np.argmax([x[measure] for x in data])
    best_epoch = idx_max + 1

    return best_epoch, data[idx_max][measure]


def read_best_model_and_info(checkpoint_name: str, best_epoch: int, device: str = 'cpu'):

    checkpoints_path = ConfigManager().get_checkpoints_path()
    mypath = os.path.join(
        checkpoints_path, checkpoint_name, f'epoch_{best_epoch}')

    map_loc = None
    if device == 'cpu':
        map_loc = torch.device('cpu')

    state_param_dir = torch.load(os.path.join(
        mypath, 'cache.pt'), map_location=map_loc)

    model_info_path = os.path.join(mypath, 'model.json')
    model_info = None
    if os.path.exists(model_info_path):
        f = open(model_info_path)
        model_info = json.load(f)
        f.close()

    opt_info_path = os.path.join(mypath, 'optimizer.json')
    optimizer_info = None
    if os.path.exists(model_info_path):
        f = open(opt_info_path)
        optimizer_info = json.load(f)
        f.close()

    return state_param_dir, model_info, optimizer_info


class PROJECT2MODE(Enum):
    ONE_VS_ONE = 0
    SILENCE_VS_REST = 1
    UNKNOWN_VS_KNOWN = 2


def project2_tune(config, criterion, device, n_trials=1, trial_name=None, n_epochs=100, mode: PROJECT2MODE = PROJECT2MODE.ONE_VS_ONE,
                  is_logging=True, epoch_logging=1, is_balanced=True, n_classes=None):

    trials = get_trials(config, n_trials)
    i = 1
    for trial in trials:
        print(f'Tuning trial: {i} / {len(trials)} - START')

        try:
            checkpoint_path = generate_checkpoint_path(trial_name)
            trial_dict = trial['trial_dict']
            trial_objects = trial['trial_objects']

            if is_logging:
                if not os.path.isdir(checkpoint_path):
                    os.makedirs(checkpoint_path)
                with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
                    json.dump(trial_dict.__str__(), f)

                with open(os.path.join(checkpoint_path, "config.pickle"), 'wb') as f:
                    pickle.dump(trial_dict, f)

            model = trial_objects['model']
            train_transform = trial_objects['train_transform']
            test_transform = trial_objects['test_transform']
            optimizer = trial_objects['optimizer']
            batch_size = trial_objects['batch_size']

            if mode == PROJECT2MODE.ONE_VS_ONE:
                train_one_vs_one(n_classes, model, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device,
                                 is_logging, epoch_logging, is_balanced, trial_name, checkpoint_path)
            elif mode == PROJECT2MODE.UNKNOWN_VS_KNOWN:
                train_unknown_vs_known(model, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device,
                                       is_logging, epoch_logging, trial_name, checkpoint_path)

            elif mode == PROJECT2MODE.SILENCE_VS_REST:
                train_silence_vs_rest(model, optimizer, criterion, train_transform, test_transform, batch_size, n_epochs, device,
                                      is_logging, epoch_logging, trial_name, checkpoint_path)

        except Exception as e:
            print(f"Exception for trial {i}: {e}")

        print(f'Tuning trial: {i} / {len(trials)} - END')
        i += 1



def predict_wav_to_vec():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h", )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    dataset_name = 'speech_recognition'
    dataset_path = os.path.join(
        ConfigManager().get_dataset_path(dataset_name))
    abs_path = ConfigManager().get_prelabels_path()
    filename_rest = f'{ConfigManager().get_now()}_onevsall.txt'
    filepath_rest = os.path.join(abs_path, filename_rest)
    svr_factory = OneVsAllDataLoadersFactory(
        dataset_path, None, None, from_file_path=filepath_rest)

    test_loader = svr_factory.get_test_loader(1)

    filenames = test_loader.dataset._walker
    predictions = predict_words(model, tokenizer, test_loader)
    return predictions, filenames


def predict_words(model, tokenizer, data):
    result = []

    for i, (batch, _) in enumerate(tqdm(data)):
        out = model(batch[0]).logits
        prediction = torch.argmax(out, dim=-1)
        transcription = tokenizer.batch_decode(prediction)[0]
        transcription = transcription.lower()
        if len(transcription) == 0:
            transcription = 'silence'
        result.append(transcription)
    return result
def get_confusion_matrix(path, final_model, unknown_model, silence_model, test_transform_silence,test_transform_unknown,test_transform_final, device, n_classes=None):

    res = predict_all(final_model, unknown_model, silence_model, test_transform_silence,test_transform_unknown,test_transform_final, 1, device, n_classes, is_valid_dataset=True)

    dataset_path = ConfigManager().get_dataset_path('speech_recognition')

    if n_classes == 31 or n_classes == 30:
        labels = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5,
                    'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'zero': 10, 'one': 11, 'two': 12, 'three': 13, 'four': 14, 'five': 15, 'six': 16, 'seven': 17, 'eight': 18, 'nine': 19,
                    'happy': 20, 'house': 21, 'cat': 22, 'wow': 23, 'marvin':24, 'bird': 25, 'bed': 26, 'tree': 27, 'dog': 28, 'sheila': 29, 'silence': 30 }
        valid_loader = Project2DataLoaderFactory(dataset_path, None, test_transform_final, with_silence=True, with_unknown=True, labels=labels).get_valid_loader(1)
    
    else:
        valid_loader =  Project2DataLoaderFactory(dataset_path, None, test_transform_final, with_silence=True, with_unknown=True).get_valid_loader(1)

    y_true = valid_loader.dataset.get_target()

    y_true = [x[1] for x in y_true]
    y_pred = [x[1] for x in res]

    disp_labels = np.unique(y_true)

    save_confusion_matrix(path, y_true, y_pred, disp_labels)





