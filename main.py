from source.project2.training_scripts import predict_all
import os
import torch
import torch.optim
import torch.nn

from source.pre_trained.efficient_net import PretrainedEff_cnn
from source.training import fit
from source.transformers import TestTransformersFactory, TrainTrasformersFactory
from source.utils.config_manager import ConfigManager
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory
from source.custom_cnn.conv1d import M5
from source.custom_cnn.conv2d import CNNSpectrogram
from source.custom_lstm.cnn_lstm import NN
from source.custom_lstm.simple_lstm import Simple_LSTM
from source.project2.training_scripts import train_unknown_vs_known


# definitions
batch_size = 64
epochs = 100

model = NN(input_size=32,hidden_size=20,num_layers=2,batch_size=batch_size,no_classes=2, device="cuda")
#model = Simple_LSTM(input_size=8000,no_classes=12, hidden_size=20,num_layers =2,batch_size = batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()

transform_train = TrainTrasformersFactory.get_transformer_spectogram()
transform_test = TestTransformersFactory.get_transformer_spectogram()


# training
dataset_name = 'speech_recognition'
dataset_path = os.path.join(
    ConfigManager().get_dataset_path(dataset_name))

loaders_factory = Project2DataLoaderFactory(
    dataset_path, transform_test=transform_test, transform_train=transform_train)

train_loader = loaders_factory.get_train_loader(batch_size=batch_size)
valid_loader = loaders_factory.get_valid_loader(batch_size=batch_size)

train_unknown_vs_known(model, optimizer=optimizer,criterion=criterion,train_transform=transform_train,test_transform=transform_test,
batch_size=batch_size,n_epochs=epochs,device = 'cuda')

