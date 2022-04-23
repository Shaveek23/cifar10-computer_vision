#imports
import os
import torch
import torch.optim
import torch.nn

from source.custom_cnn.conv1d import M5
from source.training import fit
from source.transformers import TestTransformersFactory, TrainTrasformersFactory
from source.utils.config_manager import ConfigManager
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory


# definitions
batch_size = 64
epochs = 10

model = M5()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

transform_train = TrainTrasformersFactory.get_transformer_resample()
transform_test = TestTransformersFactory.get_transformer_resample()


# training
dataset_name = 'speech_recognition'
dataset_path = os.path.join(ConfigManager().get_dataset_path(dataset_name), 'train')

loaders_factory = Project2DataLoaderFactory(dataset_path, transform_train=transform_train, transform_test=transform_test)

train_loader = loaders_factory.get_train_loader(batch_size=batch_size)
valid_loader = loaders_factory.get_valid_loader(batch_size=batch_size)


fit(model, train_loader, valid_loader, optimizer, criterion, epochs=epochs, device='cuda', is_logging=True, epoch_logging=1)
