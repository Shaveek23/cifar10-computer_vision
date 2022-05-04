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

def get_trans_spect_mel():
    return transforms.Compose([
        transforms_audio.MelSpectrogram(n_fft=512, hop_length=128, n_mels=90)
    ])

def get_trains_spect_mel_aug():
    return transforms.Compose([
        transforms_audio.MelSpectrogram(n_fft=512, hop_length=128, n_mels=90),
        torchaudio.transforms.TimeMasking(time_mask_param=10),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=50)
    ])

def get_raw_data_resampled(new_freq=8_000, norm=False):     
    if norm:
        return transforms.Compose([transforms_audio.Resample(new_freq=8_000), transforms.Normalize(0.5, 0.5)])
    return  transforms.Compose([transforms_audio.Resample(new_freq=8_000)])


class AudioTrainTrasformersFactory:

    @staticmethod
    def get_train_transformer_spectogram():
        return transforms.Compose([transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram()])


    @staticmethod  
    def get_transformer_spectogram_aug(new_freq=8000):
        return get_trans_spect_aug(new_freq)


    @staticmethod
    def get_train_transformer_spectogram_mel():
        return get_trans_spect_mel()
        
    @staticmethod
    def get_train_transfomer_spectogram_mel_aug():
        return get_trains_spect_mel_aug()

    @staticmethod
    def get_train_tranformer_resampled(new_freq=8_000, norm=False):     
        return get_raw_data_resampled(new_freq, norm)


class AudioTestTrasformersFactory:

    @staticmethod
    def get_test_transformer_spectogram():
        return transforms.Compose([transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram()])

    @staticmethod 
    def get_transformer_spectogram_aug(new_freq=8000):
        return get_trans_spect_aug(new_freq)


    @staticmethod
    def get_test_transformer_spectogram_mel():
        return get_trans_spect_mel()

    @staticmethod
    def get_test_transfomer_spectogram_mel_aug():
        return get_trains_spect_mel_aug()

    @staticmethod
    def get_test_tranformer_resampled(new_freq=8_000, norm=False):     
        return get_raw_data_resampled(new_freq, norm)