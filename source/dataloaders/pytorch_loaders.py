from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchvision
from torchvision.transforms.transforms import Compose
import os
from source.utils.config_manager import ConfigManager
import torchvision.transforms as transforms
from torch.utils.data import random_split



class PytorchDataLoaderFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: Compose, transform_test: Compose):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root=f"{ConfigManager.get_base_path()}/data/torchvision/", train=True,

                                        download=True, transform=transform)
        val_size = 5000    

        train_size = len(dataset) - val_size
        self.train_ds, self.val_ds = random_split(dataset, [train_size, val_size])



    def __str__(self):
        return f'Train_transformer:{self.train_transformer.__str__()}; Test_transformer:{self.test_transfomer.__str__()}'.format(self=self)

    def get_train_loader(self, batch_size: int):
        self.__load_train_data(self.dataset_path, self.train_transformer)
        return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=True)

    def get_train_valid_loader(self, batch_size: int):
        self.__load_train_valid_data(self.dataset_path, self.train_transformer)
        return torch.utils.data.DataLoader(self.train_valid_ds, batch_size, shuffle=True)  

    def get_valid_loader(self, batch_size: int):
        self.__load_valid_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False)

    def get_test_loader(self, batch_size: int):
        self.__load_test_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False)
        
    def __load_train_data(self, data_dir: str, transform_train: Compose):
      return self.train_ds
        #self.train_ds = torchvision.datasets.CIFAR10(root=f"{ConfigManager.get_base_path()}/data/torchvision/", train=True,
                                       # download=True, transform=transform_train)

    def __load_train_valid_data(self, data_dir: str, transform_valid_train: Compose):
         self.train_valid_ds = torchvision.datasets.CIFAR10(root=f"{ConfigManager.get_base_path()}/data/torchvision/", train=True,
                                        download=True, transform=transform_valid_train)

    def __load_valid_data(self, data_dir: str, transform_valid: Compose):
      return self.valid_ds
         #self.valid_ds = torchvision.datasets.CIFAR10(root=f"{ConfigManager.get_base_path()}/data/torchvision/", train=False,
                                       #download=True, transform=transform_valid)

    def __load_test_data(self, data_dir: str, transform_test):
          self.test_ds = torchvision.datasets.CIFAR10(root=f"{ConfigManager.get_base_path()}/data/torchvision/", train=False,
                                       download=True, transform=transform_test)
        