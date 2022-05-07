import torch
from source.custom_cnn.project2.VGGLikeParametrized import VGGLikeParametrized
from source.custom_cnn.project2.vgglike import VGGLike
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.project2.training_scripts import project2_tune, PROJECT2MODE, train_one_vs_one


for n_classes in [12, 31]:
    config = {
        'net': [
            {
                'type': VGGLikeParametrized,
                'arguments': {
                    'p_last_droput': [0.2, 0.3, 0.4, 0.5],
                    'n_output': [n_classes] # in [] !!!
                }
            },
            {
                'type': VGGLike,
                'arguments': {
                    'n_output': [n_classes] # in [] !!!
                }
            }
        ],
        'optimizer': [
            {
                'type': torch.optim.Adam,
                'arguments': {
                    'lr': [0.1, 0.01, 0.05, 0.001],
                    'weight_decay': [0, 0.001, 0.01]
                }
            }
        ],
        'transform': [
            {
                'type': {
                    'train': AudioTrainTrasformersFactory.get_train_transformer_spectogram_mel,
                    'test': AudioTrainTrasformersFactory.get_train_transformer_spectogram_mel,

                },
                'arguments': {
                    'new_freq': [8_000, 16_000]
                }
            }
        ],
        'batch_size': [32, 64, 128]
    }

    criterion = torch.nn.CrossEntropyLoss()

    project2_tune(config, criterion, 'cuda', 10, f'VGG_{n_classes}_noaug', n_epochs=100, mode=PROJECT2MODE.ONE_VS_ONE, n_classes=n_classes)
    
    
