import torch
from source.custom_cnn.project2.vgglike import VGGLike
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.transformers import TrainTrasformersFactory, TestTransformersFactory

from source.project2.training_scripts import project2_tune, PROJECT2MODE

n_classes = 12
config = {
    'net': [
        {
            'type': VGGLike,
            'arguments': {
               'p_last_droput': [0.2, 0.3, 0.4],
               'n_output': [n_classes] # in [] !!!
            }
        }
    ],
    'optimizer': [
        {
            'type': torch.optim.Adam,
            'arguments': {
                'lr': [0.01, 0.001]
            }
        }
    ],
    'transform': [
        {
            'type': {
                'train': AudioTrainTrasformersFactory.get_transformer_spectogram_aug,
                'test': AudioTestTrasformersFactory.get_transformer_spectogram_aug
            },
            'arguments': {
                'new_freq': [8_000, 16_000] 
            }
        },
        {
            'type': {
                'train': TrainTrasformersFactory.get_transformer_spectogram,
                'test': TestTransformersFactory.get_transformer_spectogram
            },
            'arguments': {
                'spect_size': [32, 64] 
            }
        }
    ],
    'batch_size': [32, 64, 128]
}

criterion = torch.nn.CrossEntropyLoss()
project2_tune(config, criterion, device='cpu', n_trials=10, trial_name='test', n_epochs=10, mode=PROJECT2MODE.ONE_VS_ONE, n_classes=12)