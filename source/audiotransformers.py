from torch.nn import Sequential
import torchaudio.transforms as transforms_audio
import torchvision.transforms as transsforms_vision


class AudioTrainTrasformersFactory:

    @staticmethod
    def get_train_transformer_spectogram():
        return Sequential(transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram())


class AudioTestTrasformersFactory:

    @staticmethod
    def get_test_transformer_spectogram():
        return Sequential(transforms_audio.Resample(new_freq=8_000), transforms_audio.Spectrogram())
