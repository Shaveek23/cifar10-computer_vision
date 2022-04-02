from argparse import ArgumentError
import torch                    
import torch.nn as nn            
import torch.nn.functional as F

from source.custom_cnn.image_classification_base import ImageClassificationBase

class DBN_cnn(ImageClassificationBase):
    def __init__(self, n_blocks=3, conv_k=None, drop_out=None, fc_k=128, fc_drop_out= 0.5,
        is_fc_drop_out=True, is_batch_norm=True,is_batch_norm_fc=True, is_drop_out=True, n_classes=10, n_chans=3, input_height=32, input_width=32):
        super(DBN_cnn, self).__init__()
        
        # Set up
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.n_chans = n_chans
        
        self.fc1_k = fc_k
        self.fc2_k = self.n_classes
        self.fc_drop_out = fc_drop_out
        self.is_fc_drop_out = is_fc_drop_out

        self.is_batch_norm_fc = is_batch_norm_fc
        self.total_size_reduction = 2 ** n_blocks
        self.input_height = input_height
        self.input_width = input_width
        self.final_width = (int)(input_width / self.total_size_reduction)
        self.final_height = (int)(input_height / self.total_size_reduction)


        self.conv_k = self.__check_conv_k(conv_k, n_blocks)
        self.is_drop_out = self.__check_is_drop_out(is_drop_out, n_blocks)
        self.is_batch_norm = self.__check_is_batch_norm(is_batch_norm, n_blocks)
        self.drop_out = self.__check_drop_out(drop_out, n_blocks)
        self.blocks_n_chans =  self.__get_blocks_n_chans(self.n_chans, self.conv_k, self.n_blocks)


        # Architecture

        self.blocks = nn.Sequential(
            *([DBN_Block(self.conv_k[i], self.drop_out[i],
                self.blocks_n_chans[i], self.is_batch_norm[i], self.is_drop_out[i]) for i in range(0, n_blocks)])
        )

        # Fully connected layers
        
        self.fc1 = nn.Linear(in_features=self.conv_k[-1] * self.final_height * self.final_width, out_features=self.fc1_k)
        
        self.fc_batch_norm = nn.BatchNorm1d(num_features=self.fc1_k)
        self.fc_drop_out = nn.Dropout(p=self.fc_drop_out)

        self.fc2 = nn.Linear(in_features=self.fc1_k, out_features=self.fc2_k)       
       

    def forward(self, x):

        x = self.blocks(x)

        # Fully connected layers
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)

        if self.is_batch_norm_fc:
            x = self.fc_batch_norm(x)
        x = F.relu(x)
        if self.is_fc_drop_out:
            x = self.fc_drop_out(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


    def __check_conv_k(self, conv_k, n_block):
        if conv_k == None:
            return [2 ** (5 + i) for i in range(0, n_block)]
        if len(conv_k) != n_block:
            raise ArgumentError(f"conv_k is of size: {len(conv_k)} expected size: {n_block}")
        return conv_k


    def __check_is_drop_out(self, is_drop_out, n_block):
        if is_drop_out == True or is_drop_out == False:
            return n_block * [is_drop_out]
        if len(is_drop_out) != n_block:
            raise ArgumentError(f"is_drop_out is of size: {len(is_drop_out)} expected size: {is_drop_out}")
        return is_drop_out
    

    def __check_is_batch_norm(self, is_batch_norm, n_block):
        if is_batch_norm == True or is_batch_norm == False:
            return n_block * [is_batch_norm]
        if len(is_batch_norm) != n_block:
            raise ArgumentError(f"is_batch_norm is of size: {len(is_batch_norm)} expected size: {is_batch_norm}")
        return is_batch_norm


    def __check_drop_out(self, drop_out, n_block):
        if drop_out == None:
            return [0.2 + 0.1 * i for i in range(0, n_block)]
        if len(drop_out) != n_block:
            raise ArgumentError(f"conv_k is of size: {len(drop_out)} expected size: {n_block}")
        return drop_out


    def __get_blocks_n_chans(self, n_chans, conv_k, n_block):
        return [n_chans, *[conv_k[i] for i in range(0, n_block - 1)]]


class DBN_Block(nn.Module):
    def __init__(self, conv_k, drop_out, n_chans, is_batch_norm=True, is_drop_out=True):
        
        super(DBN_Block, self).__init__()
        
        self.conv1_k = conv_k
        self.conv2_k = conv_k
        self.drop_out = drop_out
        self.is_batch_norm = is_batch_norm
        self.is_drop_out = is_drop_out
        self.n_chans = n_chans

        # Block architecture:
        self.conv1 = nn.Conv2d(in_channels=self.n_chans, out_channels=self.conv1_k, kernel_size=(3,3), padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.conv1_k)

        self.conv2 = nn.Conv2d(in_channels=self.conv1_k, out_channels=self.conv2_k, kernel_size=(3,3), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.conv2_k)

        self.drop_out1 = nn.Dropout2d(p=self.drop_out)


    def forward(self, x):

        # 1st convolutional layer
        x = self.conv1(x)

        # batch normalization (optional) 
        if self.is_batch_norm:
            x = self.batch_norm1(x)

        x = F.relu(x)

        # 2nd convolutional layer
        x = self.conv2(x)

        # batch normalization (optional)
        if self.is_batch_norm:
            x = self.batch_norm2(x)

        x = F.relu(x)

        # max pooling
        x = F.max_pool2d(x, (2, 2))

        # drop out (optional)
        if self.is_drop_out:
            x = self.drop_out1(x)

        return x
    