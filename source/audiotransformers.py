import torchaudio.transforms as transforms_audio 
import torchvision.transforms as transforms
import torch
import torchaudio


# definitions of common functions
def get_trans_spect_aug(new_freq=8000):
    return transforms.Compose([torch.nn.Sequential(
        torchaudio.transforms.Resample(new_freq=new_freq),
        torchaudio.transforms.Spectrogram(),
        torchaudio.transforms.TimeMasking(time_mask_param=10),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=50)
        )])


class AudioTrainTrasformersFactory:

    @staticmethod
    def get_train_transformer_spectogram():
        return transforms.Compose(transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram())


    @staticmethod  
    def get_transformer_spectogram_aug(new_freq=8000):
        return get_trans_spect_aug(new_freq)
        

class AudioTestTrasformersFactory:

    @staticmethod
    def get_test_transformer_spectogram():
        return transforms.Compose(transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram())

    @staticmethod 
    def get_transformer_spectogram_aug(new_freq=8000):
        return get_trans_spect_aug(new_freq)

