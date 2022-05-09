import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from source.custom_lstm.LSTMBase import LSTMBase


class Simple_LSTM(LSTMBase):
    def __init__(self, input_size=1, no_classes=12, hidden_size=11, num_layers=1, batch_size=64, device="cpu") -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first = True)
        self.hidden = self._init_hidden(batch_size, hidden_size, 1, num_layers,device)
        self.linear1 = nn.Linear(hidden_size, no_classes)
    def forward(self, x):
        x, (hn, cn) = self.lstm(x, self.hidden)
        y  = self.linear1(x[:, -1, :])       
        return y
