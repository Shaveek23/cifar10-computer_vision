from operator import truediv
import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from torchvision import models
from source.custom_cnn.image_classification_base import ImageClassificationBase
import pretrainedmodels


class PretrainedRESNET_cnn(ImageClassificationBase):
    def __init__(self) -> None:
        super(PretrainedRESNET_cnn,self).__init__()

        self.model =  models.resnet101(pretrained=True)
        for params in self.model.parameters():
          params.requires_grad = False
        self.num_in_features = self.model.fc.in_features
        self.num_out_features = 10

        self.model.fc = nn.Linear(in_features=self.num_in_features, out_features=120)
        self.model.fc2 = nn.Linear(120, 85)
        self.model.fc3 = nn.Linear(85, 10)
        

    def forward(self,x):
      x = self.model.forward(x)
      return x
