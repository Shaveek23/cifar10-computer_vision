import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import numpy as np
from source.custom_cnn.image_classification_base import ImageClassificationBase


class CNNSpectrogram(ImageClassificationBase):
    def __init__(self, n_input=1, n_output=11, image_height=201, image_width=41) -> None:
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(in_channels=n_input, out_channels=32,  kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same")
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same")
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same")

        # dropout layers
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)

        # linear layers
        #TODO: zparametryzowac
        self.linear1 = nn.Linear(in_features=1024, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=n_output)

        # the rest layers
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout2(x)
        x = F.log_softmax(self.linear2(x), dim = 1)
        return x
