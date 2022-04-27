import torchaudio
import torchvision
import torchvision.transforms as transforms
from source.utils.cut_out import CutOut
import random
import torch
import torchaudio


class TrainTrasformersFactory:

    @staticmethod
    def get_train_transformer():
        return torchvision.transforms.Compose([

            torchvision.transforms.ToTensor(),
            # Standardize each channel of the image
            torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.5])])

    @staticmethod
    def get_transform_with_cut_out_custom():
        return torchvision.transforms.Compose([

            torchvision.transforms.ToTensor(),
            CutOut(random.randint(1, 4), random.randint(3, 8)),
            # Standardize each channel of the image
            torchvision.transforms.Normalize([0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.5])])

    @staticmethod
    def get_transformer_kaggle():
        """
        https://www.kaggle.com/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch
        """
        return transforms.Compose([transforms.Resize((244, 244)),  # resises the image so it can be perfect for our model.
                                   transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                   # Rotates the image to a specified angel
                                   transforms.RandomRotation(10),
                                   # Performs actions like zooms, change shear angles.
                                   transforms.RandomAffine(
                                       0, shear=10, scale=(0.8, 1.2)),
                                   # Set the color params
                                   transforms.ColorJitter(
                                       brightness=0.2, contrast=0.2, saturation=0.2),
                                   transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                                   # Normalize all the images
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])

    @staticmethod
    def get_transformer_cifar_auto():
        return transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @staticmethod
    def get_transformer_only_rotation():
        return transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @staticmethod
    def get_transformer_kaggle_vgg():
        return transforms.Compose([transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                                   transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                   # Rotates the image to a specified angel
                                   transforms.RandomRotation(10),
                                   # Performs actions like zooms, change shear angles.
                                   transforms.RandomAffine(
                                       0, shear=10, scale=(0.8, 1.2)),
                                   # Set the color params
                                   transforms.ColorJitter(
                                       brightness=0.2, contrast=0.2, saturation=0.2),
                                   transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # Normalize all the images

    @staticmethod
    def get_transformer_resample():
        return torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000))
    @staticmethod
    def get_transformer_spectogram():
        return torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000),torchaudio.transforms.Spectrogram())

    @staticmethod
    def get_transformer_spectogram_eff_new():
        return transforms.Compose([torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000),
                                   torchaudio.transforms.Spectrogram()),
                                   transforms.Resize((224, 224)),
                                   transforms.Normalize((0.5), (0.5))])


class TestTransformersFactory:
    @staticmethod
    def get_transformer_efficient():
        return torchvision.transforms.Compose([
            transforms.Resize((256, 224)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

    @staticmethod
    def get_test_transformer():
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def get_transformer_kaggle():
        """
           validate on get_transformer_kaggle just normalization 
        """
        return torchvision.transforms.Compose([
            transforms.Resize((244, 244)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def get_transformer_kaggle_all():
        """
           validate on get_transformer_kaggle with trasformation
        """
        return transforms.Compose([transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                                   transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                   # Rotates the image to a specified angel
                                   transforms.RandomRotation(10),
                                   # Performs actions like zooms, change shear angles.
                                   transforms.RandomAffine(
                                       0, shear=10, scale=(0.8, 1.2)),
                                   # Set the color params
                                   transforms.ColorJitter(
                                       brightness=0.2, contrast=0.2, saturation=0.2),
                                   transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                                   # Normalize all the images
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])

    @staticmethod
    def get_transformer_resample():
        return torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000))

    @staticmethod
    def get_transformer_spectogram_eff_new():
        return transforms.Compose([torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000),
                                   torchaudio.transforms.Spectrogram()),
                                   transforms.Resize((224, 224)),
                                   transforms.Normalize((0.5), (0.5))])
    @staticmethod
    def get_transformer_spectogram():
        return torch.nn.Sequential(torchaudio.transforms.Resample(new_freq=8_000),torchaudio.transforms.Spectrogram())
