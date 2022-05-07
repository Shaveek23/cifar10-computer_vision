import torch
from source.custom_cnn.project2.vgglike import VGGLike
from source.custom_cnn.project2.VGGLikeParametrized import VGGLikeParametrized
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.project2.training_scripts import project2_tune, PROJECT2MODE

for n_classes in [31, 12, 10]:
    config = {
        'net': [
            {
                'type': VGGLike,
                'arguments': {
                'p_last_droput': [0.1, 0.2, 0.3],
                'n_output': [n_classes] 
                }
            },
            {
                'type': VGGLikeParametrized,
                'arguments': {
                'p_last_droput': [0.1, 0.2, 0.3],
                'n_output': [n_classes],
                'K': [4, 8, 16, 32] 
                }
            }
        ],
        'optimizer': [
            {
                'type': torch.optim.Adam,
                'arguments': {
                    'lr': [0.01, 0.05, 0.001],
                    'weight_decay': [0, 0.0001]
                }
            }
        ],
        'transform': [
            {
                'type': {
                    'train': AudioTrainTrasformersFactory.get_train_transformer_spectogram_mel,
                    'test': AudioTestTrasformersFactory.get_test_transformer_spectogram_mel
                },
                'arguments': {

                }
            }
        ],
        'batch_size': [32, 64, 128]
    }

    criterion = torch.nn.CrossEntropyLoss()
    
    if n_classes == 31:
        project2_tune(config, criterion, device='cuda', n_trials=10, trial_name=f'conv2d_noaug_nclass_{n_classes}', n_epochs=50, mode=PROJECT2MODE.ONE_VS_ONE, n_classes=n_classes)
    else:
        project2_tune(config, criterion, device='cuda', n_trials=10, trial_name=f'conv2d_noaug_nclass_{n_classes}', n_epochs=30, mode=PROJECT2MODE.ONE_VS_ONE, n_classes=n_classes)





