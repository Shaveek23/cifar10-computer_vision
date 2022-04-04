
from path import Path
import matplotlib.image as mpimg
import torchvision.transforms as transforms
import torch
from source.data_augementation import DataAugmentatorFactory
import os
from source.utils.cut_out import CutOut


class CutOutAugmentation(DataAugmentatorFactory):
    def __init__(self, augmentation_name,nholes =1, length = 8):
        super().__init__(augmentation_name)
        self.CutOut = CutOut(n_holes=nholes, length=length)

    def augment(self):
        self.__augment_cut_out()
        
   
    
    def __augment_cut_out(self):

        for label in os.listdir(self.image_folder):
            os.makedirs(Path.joinpath(self.images_after_augmentation_folder,label), exist_ok=True)

            for image in os.listdir(Path.joinpath(self.image_folder,label)):           
                orginal_img=mpimg.imread(Path.joinpath(self.image_folder,label,image))
                augmented_img = self.CutOut(torch.from_numpy(orginal_img)) 
                orginal_img = self.tensor_to_image(augmented_img)  
                    
                orginal_img.save(Path.joinpath(self.images_after_augmentation_folder,\
                    label,self.augmentation_name+"_"+image))
                


