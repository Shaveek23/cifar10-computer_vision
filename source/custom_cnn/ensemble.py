import torch.nn as nn
import torch.nn.functional as F
import torch
from source.custom_cnn.image_classification_base import ImageClassificationBase

class MyEnsemble(ImageClassificationBase):

    def __init__(self, models, outputs_number):
        super(MyEnsemble, self).__init__()
        self.models = nn.Sequential(
            *models
        )
        self.outputs_number = outputs_number


    def forward(self, x):
        out_total = torch.zeros(x.shape[0], self.outputs_number)
        for model in self.models:
            out = model(x)  # clone to make sure x is not changed by inplace methods
            out_total += out

        return F.log_softmax(out_total, dim=1)
