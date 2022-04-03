from operator import truediv
import torch                    
import torch.nn as nn            
import torch.nn.functional as F
from torchvision import models
from source.custom_cnn.image_classification_base import ImageClassificationBase



class PretrainedVGG16_cnn(ImageClassificationBase):
    def __init__(self) -> None:
        super(PretrainedVGG16_cnn,self).__init__()

        self.model =  models.vgg16(pretrained=True)
        num_in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_in_features,10)
        # for params in self.model.parameters():
        #   params.requires_grad = False
        

    def forward(self,x):
      x = self.model.forward(x)
      return x
