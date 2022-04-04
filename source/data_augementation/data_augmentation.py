

from source.utils.config_manager import ConfigManager
from path import Path
from PIL import Image
from abc import ABC, abstractmethod
import numpy as np


IMAGES_FOLDER = ConfigManager.get_dataset_path("cifar10")
class DataAugmentatorFactory(ABC):
    def __init__(self, augmentation_name):
        self.augmentation_name = augmentation_name
        self.image_folder = Path.joinpath(IMAGES_FOLDER,"train_valid_test\\train")
        self.images_after_augmentation_folder = Path.joinpath(IMAGES_FOLDER,augmentation_name)
        
    @abstractmethod
    def augment(self):
        pass
    @staticmethod
    def tensor_to_image(tensor):
         # source : "https://cloudxlab.com/assessment/displayslide/5658/converting-tensor-to-image"    
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
    

