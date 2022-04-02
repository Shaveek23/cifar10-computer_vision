

import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class CIFAR10_from_array(Dataset): 
    def __init__(self, data, label, images,dataset_path, transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        #self.data = torch.from_numpy(data).float()
        #self.label = torch.from_numpy(label).long()
        self.data = []
        for image in images:
            im = Image.open(os.path.join(dataset_path,'train_valid_test', 'test','unknown',image))
            self.data.append(np.asarray(im))
        # self.label = label
        self.transform = transform
        #self.img_shape = data.shape
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        
        img = Image.fromarray(self.data[index])
        #label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
            #label = torch.from_numpy(label).long()
        return img ,0
        
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.data)
    
    # def plot_image(self, number):
    #     file = self.data
    #     label = self.label
    #     fig = plt.figure(figsize = (3,2))
    #     #img = return_photo(batch_file)
    #     plt.imshow(file[number])
    #     plt.title(classes[label[number]])