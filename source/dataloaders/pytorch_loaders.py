from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchvision
from torchvision.transforms.transforms import Compose
import os
from source.utils.config_manager import ConfigManager
import torchvision.transforms as transforms
from torch.utils.data import random_split

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np 

class PytorchDataLoaderFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: Compose, transform_test: Compose):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test
       
        data_dir = f"{ConfigManager.get_base_path()}/data/torchvision/"

        self.train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        self.valid_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_test)
        self.test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

        valid_size = 0.1
        train_length = len(self.train_ds)
        indices = list(range(len(self.valid_ds)))
        split = int(np.floor(valid_size * train_length))

        np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.validation_sampler = SubsetRandomSampler(valid_idx)


    def __str__(self):
        return f'Train_transformer:{self.train_transformer.__str__()}; Test_transformer:{self.test_transfomer.__str__()}'.format(self=self)

    def get_train_loader(self, batch_size: int):
        return DataLoader(self.train_ds,batch_size=batch_size,sampler=self.train_sampler)

    def get_train_valid_loader(self, batch_size: int):
        return DataLoader(self.train_ds,batch_size=batch_size,sampler=self.train_sampler)

    def get_valid_loader(self, batch_size: int):  
        return DataLoader(self.valid_ds,batch_size=batch_size,sampler=self.validation_sampler)

    def get_test_loader(self, batch_size: int):
        return DataLoader(self.test_ds,shuffle=True,batch_size=batch_size)
             