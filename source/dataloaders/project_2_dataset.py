import os
from pathlib import Path
from typing import Tuple, Optional, Union, Dict
from torch.utils.data import Dataset

import torch
import torchaudio
from torch import Tensor
import numpy as np

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
EXRTA_SILENCE = "silence_extra"
UNKNOWN_DIRS = ["bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house", "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two", "wow", "zero"]


class Project_2_Dataset(Dataset):

    def __init__(
        self,
        with_silence: True, # True, False, 'extra'
        with_unknown: True,
        root: Union[str, Path],
        subset: Optional[str] = None,
        transform: torch.nn.Sequential = None,
        labels: Dict = None
    ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " +
            "{'training', 'validation', 'testing'}."
        )


        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._path = root
        self._transform = transform

        if labels == None:
            self._labels = {'yes': 0, 'no': 1, 'up': 2, 'down': 3, 'left': 4, 'right': 5,
                            'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10, 'silence': 11}
        else:
            self._labels = labels


        valid_filenames = ['validation_list.txt']
        test_filenames = []
        if with_silence is not False:
            valid_filenames += ['validation_silence.txt']
            test_filenames += []
        train_filenames_exlude = valid_filenames + test_filenames

        


        if subset == "validation":
            walker = self.__load_list(
                self._path, *valid_filenames)
            self._walker = [
                w
                for w in walker
                if (with_unknown or not any(unknown in w for unknown in UNKNOWN_DIRS))  
            ]
        elif subset == "testing":
            raise NotImplemented()
        elif subset == "training":
            excludes = set(self.__load_list(self._path, *train_filenames_exlude))
            walker = sorted(str(p)
                            for p in Path(self._path).glob("audio/*/*.wav"))
            self._walker = [
                w
                for w in walker
                if EXCEPT_FOLDER not in w
                and os.path.normpath(w) not in excludes
                and (with_silence or 'silence' not in w) # with_silence == (True, 'extra') -> zostają pliki z silence, w. p. p. with_silence == False -> nie ma z silence
                and (with_silence == 'extra' or EXRTA_SILENCE not in w) # with_silence == 'extra' -> wczytuje silence_extra, w. p. p. with_silence == (False, True) -> nie ma z silence_extra
                and (with_unknown or not any(unknown in w for unknown in UNKNOWN_DIRS))  
            ]
        else:
            walker = sorted(str(p)
                            for p in Path(self._path).glob("*/*.wav"))
            self._walker = [ 
                w 
                for w in walker 
                if HASH_DIVIDER in w 
                and EXCEPT_FOLDER not in w 
            ]

        return None

       

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]

        waveform, sample_rate, label = self.__load_speechcommands_item(
            fileid, self._path)
        label = self._labels[label] if label in self._labels else self._labels['unknown']

        return waveform, label

    def __len__(self) -> int:
        return len(self._walker)

    def __load_list(self, root, *filenames):
        output = []
        for filename in filenames:
            filepath = os.path.join(root, filename)
            with open(filepath) as fileobj:
                output += [os.path.normpath(os.path.join(os.path.join(
                    root, 'audio'), line.strip())) for line in fileobj]
        return output

    def __load_speechcommands_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str]:
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        label = os.path.split(label)[1]
        #speaker, _ = os.path.splitext(filename)
        #speaker, _ = os.path.splitext(speaker)

        #speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        #utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = self.__load_wav_file(filepath)

        if self._transform is not None:
            waveform, sample_rate = self.__transform(waveform, sample_rate)

        return waveform, sample_rate, label  # , speaker_id, utterance_number

    def __transform(self, waveform: Tensor, sample_rate: int) -> Tuple[Tensor, int]:

        waveform = self._transform(waveform)
        return waveform, sample_rate

    def __load_wav_file(self, filepath) -> Tuple[Tensor, int]:

        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.cpu().detach().numpy()[0]
        if len(waveform) < sample_rate:
            diff = sample_rate - len(waveform)
            left_pad = diff / 2
            right_pad = diff / 2 if diff % 2 == 0 else diff / 2 + 1
            waveform = np.pad(waveform, pad_width=(
                int(left_pad), int(right_pad)), mode='edge')

        waveform = torch.tensor(waveform)
        waveform = torch.unsqueeze(waveform, 0)
        return waveform, sample_rate
