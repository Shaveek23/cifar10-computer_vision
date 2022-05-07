import torch
from source.custom_cnn.conv1d import M5
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.project2.training_scripts import project2_tune, PROJECT2MODE, train_one_vs_one


model = M5(n_output=12, n_channel=32, stride=4)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_transform  = AudioTrainTrasformersFactory.get_train_tranformer_resampled(new_freq=16_000)
test_transform = AudioTestTrasformersFactory.get_test_tranformer_resampled(new_freq=16_000)
train_one_vs_one(12, model, opt, criterion, n_epochs=100, device='cuda', trial_name='conv1d_12_best')
