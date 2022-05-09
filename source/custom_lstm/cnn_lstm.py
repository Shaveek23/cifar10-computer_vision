import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

from source.custom_lstm.LSTMBase import LSTMBase
class NN(LSTMBase):
    def __init__(self,no_classes=12, hidden_size=11,num_layers =1,batch_size = 64,dropout_inner = 0,dropout_outter = 0, device = "cpu") -> None:
        is_binary = False
        if no_classes == 2:
            is_binary = True
        super(NN,self).__init__(is_binary)     

        self.lstm = nn.LSTM(32, hidden_size, num_layers,batch_first = True, dropout = dropout_inner)
        self.hidden = self._init_hidden(batch_size, hidden_size, 1, num_layers,device)
        self.linear1 = nn.Linear(hidden_size, no_classes)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16, kernel_size=5,stride = 2,padding = 2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=5,stride = 2,padding = 2)
        self.dropout = nn.Dropout(p=dropout_outter)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)        
        x, (hn, cn) = self.lstm(x, self.hidden)
        x = self.dropout(x)
        y  = self.linear1(x[:, -1, :])       
        return y

