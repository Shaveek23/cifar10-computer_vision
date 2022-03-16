import torchvision
import torchvision.transforms as transforms

class TrainTrasformersFactory:

    @staticmethod
    def get_train_transformer():
        return torchvision.transforms.Compose([
            # Scale the image up to a square of 40 pixels in both height and width
            torchvision.transforms.Resize(40),
            # Randomly crop a square image of 40 pixels in both height and width to
            # produce a small square of 0.64 to 1 times the area of the original
            # image, and then scale it to a square of 32 pixels in both height and
            # width
            torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
            ratio=(1.0, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # Standardize each channel of the image
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
            [0.2023, 0.1994, 0.2010])])


    @staticmethod
    def get_transformer_kaggle():
        """
        https://www.kaggle.com/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch
        """
        return transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])


class TestTransformersFactory:

    @staticmethod
    def get_test_transformer():
        return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010])])


    def get_transformer_kaggle():
        """
           validate on get_transformer_kaggle just normalization 
        """
        return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_transformer_kaggle_all():
        """
           validate on get_transformer_kaggle with trasformation
        """
        return transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
