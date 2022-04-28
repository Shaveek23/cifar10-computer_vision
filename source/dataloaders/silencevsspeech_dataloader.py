from xmlrpc.client import Boolean
from source.dataloaders.data_loaders_factory import DataLoaderFactory
import torch
import torchaudio
import os
import numpy as np

from source.dataloaders.project_2_dataset import Project_2_Dataset


class SilenceVsSpeechDataLoadersFactory(DataLoaderFactory):

    def __init__(self, dataset_path, transform_train: torch.nn.Sequential = None, transform_test: torch.nn.Sequential = None):
        super().__init__(dataset_path)
        self.train_transformer = transform_train
        self.test_transfomer = transform_test
        self.labels = {'silence' : 0, 'silence_extra': 0, 'unknown': 1 }
        

    def __str__(self):
        return f'Silence vs. speech: Train_transformer:{self.train_transformer.__str__()}; Silence vs. speech: Test_transformer:{self.test_transfomer.__str__()}'.format(self=self)


    def get_train_loader(self, batch_size: int):
        self.__load_train_data(
            self.dataset_path, self.train_transformer)

        sampler = self.__get_sampler_to_balance_classes(self.train_ds._walker)
        return torch.utils.data.DataLoader(self.train_ds, batch_size, shuffle=False, drop_last=True, sampler = sampler)


    def get_train_valid_loader(self, batch_size: int):
        raise NotImplemented()


    def get_valid_loader(self, batch_size: int):
        self.__load_valid_data(
            self.dataset_path, self.test_transfomer)

        sampler = self.__get_sampler_to_balance_classes(self.valid_ds._walker)
        return torch.utils.data.DataLoader(self.valid_ds, batch_size, shuffle=False, drop_last=True, sampler=sampler)


    def get_test_loader(self, batch_size: int):
        self.__load_test_data(
            self.dataset_path, self.test_transfomer)
        
        return torch.utils.data.DataLoader(self.test_ds, batch_size, shuffle=False, drop_last=False)


    def __load_train_data(self, data_dir: str, transform_train: torch.nn.Sequential):
        self.train_ds = Project_2_Dataset(
            with_silence='extra', with_unknown=True, root=data_dir, subset='training', transform=transform_train, labels=self.labels)


    def __load_valid_data(self, data_dir: str, transform_valid: torch.nn.Sequential):
        self.valid_ds = Project_2_Dataset(
              with_silence='extra', with_unknown=True, root=data_dir, subset='validation', transform=transform_valid, labels=self.labels)


    def __load_test_data(self, data_dir: str, transform_test: torch.nn.Sequential):
        self.test_ds = Project_2_Dataset(
            True, True, data_dir, 'testing', transform=transform_test)


    def __get_sampler_to_balance_classes(self, samples_filenames):
        zeros_for_silence = [0 if 'silence' in x else 1 for x in samples_filenames ]
        samples_count = len(samples_filenames)
        non_silence_count = np.count_nonzero(zeros_for_silence)
        silence_count = samples_count - non_silence_count 

        class_sample_count = [silence_count,  non_silence_count]
        weights = 1 / torch.Tensor(class_sample_count)
        weights = weights.double()
        sample_weights = weights[zeros_for_silence]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler
