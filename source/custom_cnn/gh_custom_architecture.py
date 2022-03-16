
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
BATCH_SIZE = 4

class GH_CNN(nn.Module):
    # def __init__(self):
    #     super(GH_CNN, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 4 * 4, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # print(self.size_1_cnn)
        # print(self.size_1_pooling)
        # print(self.size_2_cnn)
        # print(self.size_2_pooling)


        x=  self.conv1(x)
        #input(f'{x.shape}')
        x = F.relu(x)
        #input(f'{x.shape}')
        x = self.pooling1(x)
        #input(f'{x.shape}')
        x = self.conv2(x)
        #input(f'{x.shape}')
        x = F.relu(x)
        #input(f'{x.shape}')
        x=self.pooling2(x)
        #input(f'{x.size}')
        x = x.view(-1,int(self.size_2_pooling)*int(self.size_2_pooling)*self.channel2_out)
        x = self.fc1(x)
        #input(f'{x.size}')
        # x = nn.Sigmoid(x)
        # input(f'{x.size}')
        x = F.relu(x)
        x = self.fc2(x)
        #input(f'{x.size}')
        # x = nn.Sigmoid(x)
        # input(f'{x.size}')
        x = F.relu(x)
        _= self.fc3(x)
        return _ 


    def __init__(self):
        super(GH_CNN,self).__init__()
        # functions

        def count_size_after_layer(input_size, kernel_size,stride_size=1,padding_size=0):
            """
            We assume that each window is a square window
            WE NEED TO ADD DILITATION
            """
            return ((input_size+padding_size*2-kernel_size)/stride_size)+1
        
       

        # channels
        self.channel1_in = 3
        self.channel1_out = 6
        self.channel2_out = 16

        # kernels
        self.kernel1_size = 5
        self.kernel2_size = 3

        # padding
        self.padding1 = 0
        self.padding2 = 0

        # stride
        self.stride1 = 1
        self.stride2 = 1

        #dialation
        self.dilation1 = 1 
        self.dilation2 = 1

        # pooling
        self.pooling1_size = 2
        self.pooling2_size = 2

        # linear layer parameter
        self.fc1_size = 120
        self.fc2_size = 84
        self.fc3_size = 10

        # sizes
        self.size_1_cnn = count_size_after_layer(IMAGE_SIZE,self.kernel1_size,self.stride1,self.padding1) 
        
        self.size_1_pooling = self.size_1_cnn/self.pooling1_size 
        
        self.size_2_cnn = count_size_after_layer(self.size_1_pooling,self.kernel2_size,self.stride2,self.padding1)
        
        self.size_2_pooling = self.size_2_cnn/self.pooling2_size
        

        # layers
        
        self.conv1 =nn.Conv2d(in_channels = self.channel1_in, out_channels = self.channel1_out,kernel_size=(self.kernel1_size,self.kernel1_size),
        stride=self.stride1,padding=self.padding1,dilation= self.dilation1)
        self.conv2 = nn.Conv2d(in_channels = self.channel1_out, out_channels = self.channel2_out,kernel_size=(self.kernel2_size,self.kernel2_size),
        stride=self.stride2,padding=self.padding2,dilation= self.dilation2)
        self.fc1 = nn.Linear(int(self.size_2_pooling)*int(self.size_2_pooling)*self.channel2_out,self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size,self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size,self.fc3_size)
        self.pooling1 = nn.MaxPool2d(self.pooling1_size,self.pooling1_size)
        self.pooling2 = nn.MaxPool2d(self.pooling2_size,self.pooling2_size)
       

        # loss function
        #self.loss_fn = torch.nn.CrossEntropyLoss()


        


        
            

