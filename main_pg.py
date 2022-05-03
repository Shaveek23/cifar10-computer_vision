from source.project2.training_scripts import train_one_vs_one, train_silence_vs_rest, train_unknown_vs_known
import os
import torch
import torch.optim
import torch.nn

from source.training import fit
from source.audiotransformers import AudioTestTrasformersFactory, AudioTrainTrasformersFactory
from source.utils.config_manager import ConfigManager
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory
from source.custom_cnn.project2.vgglike import VGGLike
from source.project2.training_scripts import train_unknown_vs_known


# definitions
batch_size = 64
epochs = 100

transform_train = AudioTrainTrasformersFactory.get_train_transformer_spectogram_mel()
transform_test = AudioTestTrasformersFactory.get_test_transformer_spectogram_mel()

dataset_name = 'speech_recognition'
dataset_path = os.path.join(
    ConfigManager().get_dataset_path(dataset_name))

for p in [0.2, 0.5]:
## 12 classes
    model = VGGLike(n_output=12, p_last_droput=p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    train_one_vs_one(12, model, optimizer, criterion, transform_train, transform_test, batch_size, epochs, 'cuda', is_balanced=True, trial_name=f'12classes_specaug_mel_p{p}')


    # 10 classes
    model = VGGLike(n_output=10, p_last_droput=p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    train_one_vs_one(10, model, optimizer, criterion, transform_train, transform_test, batch_size, epochs, 'cuda', trial_name=f'10classes_specaug_mel{p}')


    # unknown vs known
    model = VGGLike(n_output=2, p_last_droput=p)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    train_unknown_vs_known(model, optimizer=optimizer,criterion=criterion,train_transform=transform_train,test_transform=transform_test,
    batch_size=batch_size, n_epochs=epochs, device='cuda', trial_name=f'unknownvsknown_specaug_mel{p}')


    # silence vs rest
    model = VGGLike(n_output=2, p_last_droput=p)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    train_silence_vs_rest(model, optimizer, criterion, transform_train, transform_test, batch_size, epochs, 'cuda', trial_name=f'silencevsrest_specaug_mel{p}')




