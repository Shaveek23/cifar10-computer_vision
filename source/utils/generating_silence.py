import pandas as pd
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from glob import glob
import os
import torch
import random
import sys

BACKGROUND_NOISE_DIRECTORY = os.path.join("..","..","data\\audio\\train\\audio", "_background_noise_\\")
SILENCE_DIRECTORY = os.path.join( "..","..","data\\audio\\train\\audio", "silence\\")
TXT_DIRECTORY = os.path.join("..","..","data\\audio\\train\\")
FORMAT = "wav"

possible_transformations_number = 4

def save_files(validation, test):
    pd.Series(validation).to_csv(TXT_DIRECTORY+"validation_silence.txt",
                                 header=None, index=None, sep='\n', mode='a')
    pd.Series(test).to_csv(TXT_DIRECTORY+"test_silence.txt",
                           header=None, index=None, sep='\n', mode='a')

def generate_silence(size, validation_size=0.1, test_size=0.1):
    """
    need to provide the size that's possible to be divided by 6 
    """
    files_names = []
    isExist = os.path.exists(SILENCE_DIRECTORY)
    if not isExist:
        os.makedirs(SILENCE_DIRECTORY)
    per_file_size = int(size/6)
    for file_nr, file in enumerate(glob(BACKGROUND_NOISE_DIRECTORY + "*.wav")):
        for i in range(per_file_size):
            samples, sample_rate = torchaudio.load(file)
            start = np.random.randint(len(samples[0]) - sample_rate, size=1)[0]
            cutted = samples[:, start:start+sample_rate]
            transformation = np.random.randint(
                possible_transformations_number, size=1)[0]

            if transformation == 0:
                transform = transforms.Fade(
                    fade_in_len=0, fade_out_len=sample_rate, fade_shape='linear')
                cutted = transform(cutted)
            elif transformation == 1:
                transform = transforms.Vol(10)
                cutted = transform(cutted)

            elif transformation == 2:
                noise = torch.tensor(np.random.normal(0, .05, cutted.shape))
                cutted = torch.add(noise, cutted).to(dtype=torch.float32)
            file_name = f"{file_nr}_{i}_{transformation}.{FORMAT}"
            files_names.append(f"silence/{file_name}")
            torchaudio.save(os.path.join(SILENCE_DIRECTORY,
                            file_name), cutted, sample_rate, format=FORMAT)
    files_names = np.array(files_names)
    indexes = random.sample(list(range(len(files_names))), int(
        (validation_size + test_size) * size))
    validation_indexes = indexes[: int(validation_size * size)]
    test_indexes = indexes[int(validation_size * size):]
    save_files(files_names[validation_indexes], files_names[test_indexes])

if __name__ == '__main__':
    generate_silence(int(sys.argv[1]))
    