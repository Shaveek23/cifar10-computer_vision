import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import numpy as np
from source.custom_cnn.image_classification_base import ImageClassificationBase


class VGGLike(ImageClassificationBase):
    def __init__(self, n_input=1, n_output=11) -> None:
        
        is_binary = False
        if n_output == 2:
            is_binary = True

        super(VGGLike, self).__init__(is_binary, pos_label=0)

        self.conv1 = nn.Conv2d(in_channels=n_input, out_channels=8, kernel_size=(2,2), padding='valid')

        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2,2), padding='valid')

        self.batch_norm2 = nn.BatchNorm2d(num_features=8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2,2), padding='valid')

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding='valid')

        self.batch_norm3 = nn.BatchNorm2d(num_features=16)
        self.dense1 = nn.Linear(in_features=16 * 9 * 13, out_features=32)
        self.dense2 = nn.Linear(in_features=32, out_features=n_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
       
        x = F.max_pool2d(x, (2, 2))

        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, (2, 2))

        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = F.dropout2d(x, p=0.2)

        x = self.conv4(x)
        x = F.relu(x)

        x = F.max_pool2d(x, (2, 2))
        x = self.batch_norm3(x)
        x = F.dropout2d(x, p=0.5)

        outputs_size = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1,  outputs_size)

        
        x = self.dense1(x)
        x = F.relu(x)

        x = self.dense2(x)

        return x
