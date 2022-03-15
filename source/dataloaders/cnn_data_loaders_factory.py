from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchvision
from torchvision.transforms.transforms import Compose
import os

class CNNDataLoaderFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: Compose, transform_test: Compose):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test

    def get_train_loader(self, batch_size: int):
        self.__load_train_data(self.dataset_path, self.train_transformer)
        return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=True, drop_last=True)

    def get_train_valid_loader(self, batch_size: int):
        self.__load_train_valid_data(self.dataset_path, self.train_transformer)
        return torch.utils.data.DataLoader(self.train_valid_ds, batch_size, shuffle=True, drop_last=True)  

    def get_valid_loader(self, batch_size: int):
        self.__load_valid_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False, drop_last=True)

    def get_test_loader(self, batch_size: int):
        self.__load_test_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False, drop_last=False)
        
    def __load_train_data(self, data_dir: str, transform_train: Compose):
        self.train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train'),
            transform=transform_train)

    def __load_train_valid_data(self, data_dir: str, transform_valid_train: Compose):
         self.train_valid_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train_valid'),
            transform=transform_valid_train)

    def __load_valid_data(self, data_dir: str, transform_valid: Compose):
         self.valid_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', 'valid'),
            transform=transform_valid)

    def __load_test_data(self, data_dir: str, transform_test):
          self.test_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', 'test'),
            transform=transform_test)
        