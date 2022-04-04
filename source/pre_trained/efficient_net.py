from operator import truediv
import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from torchvision import models
from source.custom_cnn.image_classification_base import ImageClassificationBase
import pretrainedmodels


class PretrainedEff_cnn(ImageClassificationBase):
    def __init__(self) -> None:
        super(PretrainedEff_cnn,self).__init__()

        self.model =  models.efficientnet_b0(pretrained=True)
        # for params in self.model.parameters():
        #   params.requires_grad = False
    
        self.model.classifier[1]=nn.Linear(in_features=1280,out_features= 10)
        

    def forward(self,x):
      x = self.model.forward(x)
      return x
