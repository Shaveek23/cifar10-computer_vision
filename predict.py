from source.project2.training_scripts import predict_all
import os
import torch
import torch.optim
import torch.nn

from source.pre_trained.efficient_net import PretrainedEff_cnn
from source.training import fit
from source.audiotransformers import AudioTestTrasformersFactory,AudioTrainTrasformersFactory
from source.utils.config_manager import ConfigManager
from source.dataloaders.project_2_dataloaders_factory import Project2DataLoaderFactory
from source.custom_cnn.conv1d import M5
from source.custom_cnn.conv2d import CNNSpectrogram
from source.custom_lstm.cnn_lstm import NN
from source.custom_lstm.simple_lstm import Simple_LSTM
from source.custom_cnn.project2.vgglike import VGGLike
from source.utils.submission_creator import create_kaggle_submision_file_audio


batch_size = 2
n_classes = 31
output_file_name = "LSTM31_aug"
final_model = NN(hidden_size=20,num_layers=2,batch_size=batch_size,no_classes=n_classes, device="cuda", dropout_inner=0, dropout_outter=0)
final_model_path = os.path.join(ConfigManager().get_models_path(),"LSTM31_aug.pt")
silence_model= None
silence_model_path = ""
unknown_model = None
unknown_model_path = ""

test_transform = AudioTestTrasformersFactory.get_test_transformer_spectogram()

create_kaggle_submision_file_audio(os.path.join(ConfigManager().get_base_path(),"predictions/{output_file_name}.csv"),test_transform, final_model, final_model_path, unknown_model, unknown_model_path, 
silence_model, silence_model_path, batch_size=2, device="cuda",n_classes = n_classes)