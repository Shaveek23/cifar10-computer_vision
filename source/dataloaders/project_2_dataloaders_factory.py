from xmlrpc.client import Boolean
from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchaudio
import os

from source.dataloaders.project_2_dataset import Project_2_Dataset

class Project2DataLoaderFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: torch.nn.Sequential = None, transform_test: torch.nn.Sequential = None,
        with_silence: Boolean = True, with_unknown: Boolean = True, is_balanced=False, labels=None):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test
        self.with_silence = with_silence
        self.with_unknown = with_unknown
        self.is_balanced = is_balanced
        self.labels = labels


    def __str__(self):
        return f'Train_transformer:{self.train_transformer.__str__()}; Test_transformer:{self.test_transfomer.__str__()}'.format(self=self)


    def get_train_loader(self, batch_size: int):
        self.__load_train_data(
            self.dataset_path, self.train_transformer)

        if self.is_balanced:
            sampler = self.__get_sampler_to_balance_classes(self.train_ds)
            return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=False, drop_last=True, sampler=sampler)

        return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=True, drop_last=True)


    def get_train_valid_loader(self, batch_size: int):
        raise NotImplemented()


    def get_valid_loader(self, batch_size: int):
        self.__load_valid_data(
            self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False, drop_last=True)


    def get_test_loader(self, batch_size: int):
        self.__load_test_data(
            self.dataset_path, self.test_transfomer)
        return torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False, drop_last=True)


    def __load_train_data(self, data_dir: str, transform_train: torch.nn.Sequential):
        self.train_ds = Project_2_Dataset(
            self.with_silence, self.with_unknown, data_dir, 'training', transform=transform_train, labels=self.labels)


    def __load_valid_data(self, data_dir: str, transform_valid: torch.nn.Sequential):
        self.valid_ds = Project_2_Dataset(
            self.with_silence, self.with_unknown, data_dir, 'validation', transform=transform_valid, labels=self.labels)


    def __load_test_data(self, data_dir: str, transform_test: torch.nn.Sequential):
        self.test_ds = Project_2_Dataset(
            self.with_silence, self.with_unknown, data_dir, 'testing', transform=transform_test)


    def __get_sampler_to_balance_classes(self, dataset):
       
        class_sample_count = torch.zeros(50, dtype=torch.long)
        for _, target in dataset:
            class_sample_count[target] += 1

        nonZeroRows = class_sample_count > 0
        class_sample_count = class_sample_count[nonZeroRows]

        weights = 1 / class_sample_count
        weights = weights.double()

        sample_weights = []
        for _, target in dataset:                                          
            sample_weights.append(weights[target].item())  

        sample_weights = torch.tensor(sample_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler
