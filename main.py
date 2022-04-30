# imports
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
import sys


def run_training(dropout_inner, dropout_outter, hidden_size, withAug):
    # definitions
    batch_size = 64
    epochs = 50

    #model = DBN_cnn(n_blocks =3, n_classes = 12, n_chans= 1, input_width= 41, input_height= 201)
    model = NN(input_size=32, no_classes=12, hidden_size=hidden_size, num_layers=2,
               batch_size=batch_size, device="cuda", dropout_inner=dropout_inner, dropout_outter=dropout_outter)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    if withAug:
        transform_train = TrainTrasformersFactory.get_transformer_spectogram_aug()
    else:
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

    fit(model, train_loader, valid_loader, optimizer, criterion,
        epochs=epochs, device='cuda', is_logging=True, epoch_logging=1)


if __name__ == '__main__':
    dropout_inner = int(sys.argv[1])
    dropout_outter = int(sys.argv[2])
    hidden_size = int(sys.argv[3])
    withAug = bool(sys.argv[4])
    run_training(dropout_inner, dropout_outter, hidden_size, withAug)
