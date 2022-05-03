import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

from source.custom_cnn.image_classification_base import ImageClassificationBase

class LSTMBase(ImageClassificationBase):
    def __init__(self,binary_classification) -> None:
        super().__init__(binary_classification)
    
    def _init_hidden(self, batch_size, hidden_size, num_directions, num_layers, device):
        h1 = torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size))
        h2 = torch.autograd.Variable(torch.zeros(num_directions*num_layers, batch_size, hidden_size))
        h1,h2 = h1.to(device), h2.to(device)
        return (h1,h2)