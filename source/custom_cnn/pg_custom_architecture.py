import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

class PG_CNN(nn.Module):
    
    def __init__(self):
        super(PG_CNN, self).__init__()

        # defining the architecture

        # 1st conv layer
        self.conv1_K = 6 # number of filters (aka kernels) 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.conv1_K, kernel_size=(5,5), padding=1)
        # in_channels <- each picture has 3 channels (3 maps)
        # out_channels <- number of kernels, each kernel has a depth of 3 (in_channels), so in this case we apply 
        #                 6 kernels of size (5, 5, 3) on 3 maps and get 6 feature maps -
        #                 applying one kernel of size (5, 5, 3) on 3 maps (in_channels) results in 3 feature maps that are
        #                 summed up to make only one feature map, so after 6 kernels of size (5, 5, 3) we obtain exacly 6 feature maps


        # 2nd conv layer
        self.conv2_K = 12 # 12 filters of depth 6: [4, 4, 6] giving 12 feature maps [H,W,1]
        self.conv2 = nn.Conv2d(in_channels=self.conv1_K, out_channels=self.conv2_K, kernel_size=(4,4))

        # fully connected layers
        self.fc1 = nn.Linear(in_features=self.conv2_K*6*6, out_features=3*120)
        self.fc2 = nn.Linear(in_features=3*120, out_features=3*84)
        self.fc3 = nn.Linear(in_features=3*84, out_features=10) # output layer, it has the same number of neurons as the number of the classes 

    def forward(self, x):
        
        #INPUT: x.shape = [4, 3, 32, 32]
        
        # x.shape = [4, 3, 32, 32]
        x = self.conv1(x)
        # x.shape = [4, 6, 30, 30]
       
        # x.shape = [4, 6, 30, 30]
        x = F.relu(x)
        # x.shape = [4, 6, 30, 30]

        # x.shape = [4, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))
        # x.shape = [4, 6, 15, 15]

        # x.shape = [4, 6, 15, 15]
        x = self.conv2(x)
        # x.shape = [4, 12, 12, 12]

        # x.shape = [4, 12, 12, 12]
        x = F.relu(x)
        # x.shape = [4, 12, 12, 12]

        # x.shape = [4, 12, 12, 12]
        x = F.max_pool2d(x, 2)
        # x.shape = [4, 12, 6, 6]
      
        # x.shape = [4, 12, 6, 6]
        x = x.view(-1, self.conv2_K*6*6) # 6x6 is a size of our image
        # x.shape = [4, 12*6*6]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
