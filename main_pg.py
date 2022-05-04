import torch
from source.custom_cnn.conv1d import M5
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.project2.training_scripts import project2_tune, PROJECT2MODE

for n_classes in [2, 10, 12, 31]:
    config = {
        'net': [
            {
                'type': M5,
                'arguments': {
                'stride': [4, 8, 16, 32],
                'n_channel': [16, 32, 64],
                'n_output': [n_classes] # in [] !!!
                }
            }
        ],
        'optimizer': [
            {
                'type': torch.optim.Adam,
                'arguments': {
                    'lr': [0.01, 0.05, 0.001],
                    'weight_decay': [0, 0.001]
                }
            }
        ],
        'transform': [
            {
                'type': {
                    'train': AudioTrainTrasformersFactory.get_train_tranformer_resampled,
                    'test': AudioTestTrasformersFactory.get_test_tranformer_resampled
                },
                'arguments': {
                    'new_freq': [8_000, 16_000],
                    'is_norm': [True, False] 
                }
            }
        ],
        'batch_size': [32, 64]
    }

    criterion = torch.nn.CrossEntropyLoss()
    
    if n_classes > 2:
        project2_tune(config, criterion, device='cpu', n_trials=10, trial_name=f'conv1d_nclass_{n_classes}', n_epochs=10, mode=PROJECT2MODE.ONE_VS_ONE, n_classes=n_classes)
    else:
        project2_tune(config, criterion, device='cpu', n_trials=10, trial_name=f'conv1d_silencevsrest', n_epochs=10, mode=PROJECT2MODE.SILENCE_VS_REST, n_classes=n_classes)
        project2_tune(project2_tune(config, criterion, device='cpu', n_trials=10, trial_name=f'conv1d_knownvsunknown', n_epochs=10, mode=PROJECT2MODE.UNKNOWN_VS_KNOWN, n_classes=n_classes))



