import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

from source.custom_cnn.image_classification_base import ImageClassificationBase

class NN(ImageClassificationBase):
    def __init__(self,input_size=1,no_classes=12, hidden_size=11,num_layers =1,batch_size = 64, device = "cpu") -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first = True)
        self.hidden = self._init_hidden(batch_size, hidden_size, 1, num_layers,device)
        self.linear1 = nn.Linear(hidden_size, no_classes)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16, kernel_size=5,stride = 2,padding = 2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=5,stride = 2,padding = 2)

    def _init_hidden(self, batch_size, hidden_size, num_directions, num_layers, device):
        h1 = torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size))
        h2 = torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size))
        h1,h2 = h1.to(device), h2.to(device)
        return (h1,h2)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)        
        x, (hn, cn) = self.lstm(x, self.hidden)
        y  = self.linear1(x[:, -1, :])       
        return y

