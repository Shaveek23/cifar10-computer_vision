from operator import truediv
import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from torchvision import models
from source.custom_cnn.image_classification_base import ImageClassificationBase


class PretrainedEff_cnn(ImageClassificationBase):
    def __init__(self,number_of_classes) -> None:
        super(PretrainedEff_cnn,self).__init__()

        self.model =  models.efficientnet_b0(pretrained=True) 
        conv_weight = self.model.features[0][0].weight
        self.model.features[0][0].in_channels = 1  
        self.model.features[0][0].weight = torch.nn.Parameter(conv_weight.sum(dim = 1, keepdim=True))
        self.model.classifier[1]=nn.Linear(in_features=1280,out_features= number_of_classes)
        

    def forward(self,x):
      x = self.model.forward(x)
      return x
