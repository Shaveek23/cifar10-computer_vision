import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import numpy as np
from source.custom_cnn.image_classification_base import ImageClassificationBase



class VGGLikeParametrized(ImageClassificationBase):
    def __init__(self, n_input=1, n_output=11, K=16, p_last_droput=0.2) -> None:
        
        is_binary = False
        if n_output == 2:
            is_binary = True

        super(VGGLikeParametrized, self).__init__(is_binary, pos_label=0)

        self.p_last_dropout = p_last_droput
        self.K = K

        self.conv1 = nn.Conv2d(in_channels=n_input, out_channels=self.K, kernel_size=(2,2), padding='valid')

        self.batch_norm1 = nn.BatchNorm2d(num_features=self.K)
        self.conv2 = nn.Conv2d(in_channels=self.K, out_channels=self.K*2, kernel_size=(2,2), padding='valid')

        self.batch_norm2 = nn.BatchNorm2d(num_features=self.K*2)
        self.conv3 = nn.Conv2d(in_channels=self.K*2, out_channels=self.K*4, kernel_size=(2,2), padding='valid')

        self.conv4 = nn.Conv2d(in_channels=self.K*4, out_channels=self.K*8, kernel_size=(3,3), padding='valid')

        self.batch_norm3 = nn.BatchNorm2d(num_features=self.K*8)
        self.dense1 = nn.Linear(in_features=self.K * 8 * 9 * 13, out_features=32)
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

        x = F.adaptive_avg_pool2d(x, output_size=(13, 9))
        x = self.batch_norm3(x)
        x = F.dropout2d(x, p=self.p_last_dropout)

        input_dense1 = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, input_dense1)

        x = self.dense1(x)
        x = F.relu(x)

        x = self.dense2(x)

        return x
        