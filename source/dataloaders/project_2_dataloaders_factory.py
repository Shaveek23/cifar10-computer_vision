from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchaudio
import os

from source.dataloaders.project_2_dataset import Project_2_Dataset

class Project2DataLoaderFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: torch.nn.Sequential = None, transform_test: torch.nn.Sequential = None):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test
         

    def __str__(self):
        return f'Train_transformer:{self.train_transformer.__str__()}; Test_transformer:{self.test_transfomer.__str__()}'.format(self=self)


    def get_train_loader(self, batch_size: int):
        self.__load_train_data(self.dataset_path, self.train_transformer) 
        return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=True, drop_last=True)


    def get_train_valid_loader(self, batch_size: int):
        raise NotImplemented()


    def get_valid_loader(self, batch_size: int):
        self.__load_valid_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False, drop_last=True)


    def get_test_loader(self, batch_size: int):
        self.__load_test_data(self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False, drop_last=False)

    
    def __load_train_data(self, data_dir: str, transform_train: torch.nn.Sequential):
        self.train_ds = Project_2_Dataset(data_dir, 'training', transform=transform_train)


    def __load_valid_data(self, data_dir: str, transform_valid: torch.nn.Sequential):
         self.valid_ds = Project_2_Dataset(data_dir, 'validation', transform=transform_valid)


    def __load_test_data(self, data_dir: str, transform_test: torch.nn.Sequential):
        self.test_ds = Project_2_Dataset(data_dir, 'testing', transform=transform_test)
