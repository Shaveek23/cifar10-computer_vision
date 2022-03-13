import collections
import math
import os
import shutil
import tarfile
from urllib import request
import zipfile
from source.utils.config_manager import ConfigManager

import torch
import torchvision
from torchvision.transforms.transforms import Compose

DATA_HUB = ConfigManager.get_DATA_HUB()

def download_extract(name, folder=None): #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        raise Exception('Only .zip, .tar or .gz can be handled.')
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download(name, cache_dir=os.path.join('..', 'data')): #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    
    url = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
            if data:
                return fname
               
        print(f'Downloading {fname} from {url}...')
        r = request.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname

def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
    
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
        'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))                                   
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_data(data_dir, valid_ratio):
    
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

def get_data(data_dir, transform_train: Compose, transform_test: Compose, batch_size: int):
    """This function loads data assuming the folder data_dir exists and contains folders: train_valid_test (with train, test, valid folders) and a trainLabels.csv file."""
   
    train_ds, train_valid_ds, valid_ds, test_ds = load_data(data_dir, transform_train, transform_test)
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
        ) for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
    return train_iter, train_valid_iter, valid_iter, test_iter

def load_data(data_dir: str, transform_train: Compose, transform_test: Compose):

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]
    
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    return train_ds, train_valid_ds, valid_ds, test_ds

    