from abc import ABC, abstractmethod
class DataLoaderFactory(ABC):
   
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
   
    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_valid_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @abstractmethod
    def get_train_valid_loader(self):
        pass


   
