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


class TestTransformersFactory:

    @staticmethod
    def get_test_transformer():
        return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010])])