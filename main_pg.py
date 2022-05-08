import torch
from source.custom_cnn.project2.VGGLikeParametrized import VGGLikeParametrized
from source.custom_cnn.project2.vgglike import VGGLike
from source.audiotransformers import AudioTrainTrasformersFactory, AudioTestTrasformersFactory
from source.project2.training_scripts import project2_tune, PROJECT2MODE, train_one_vs_one


model = VGGLikeParametrized(n_output=31, p_last_droput=0.3)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
train_transform  = AudioTrainTrasformersFactory.get_train_transformer_spectogram_mel()
test_transform = AudioTestTrasformersFactory.get_test_transformer_spectogram_mel()

train_one_vs_one(31, model, opt, criterion, train_transform, test_transform, batch_size=32, n_epochs=100, device='cuda', trial_name='conv2d_31_vgglikeparametrized_p03')
    