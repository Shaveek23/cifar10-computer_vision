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



class Project_2_Dataset(Dataset):

    def __init__(
            self,
            root: Union[str, Path],
            subset: Optional[str] = None,
            transform: torch.nn.Sequential = None,
            labels: Dict = None
        ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " + "{'training', 'validation', 'testing'}."
        )

         # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._path = root
        self._transform = transform

        if labels == None:
            self._labels = {'yes' : 0, 'no' : 1, 'up' : 2, 'down': 3, 'left': 4, 'right': 5, 'on': 6, 'off': 7, 'stop': 8, 'go': 9, 'unknown': 10, 'silence': 11 }
        else:
            self._labels = labels
    

        if subset == "validation":
            self._walker = self.__load_list(self._path, "validation_list.txt")
        elif subset == "testing":
            self._walker = self.__load_list(self._path, "testing_list.txt")
        elif subset == "training":
            excludes = set(self.__load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob("audio/*/*.wav"))
            self._walker = [
                w
                for w in walker
                if EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

        
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]

        waveform, sample_rate, label = self.__load_speechcommands_item(fileid, self._path)
        label = self._labels[label] if label in self._labels else self._labels['unknown']
        
        return waveform, label


    def __len__(self) -> int:
        return len(self._walker)


    def __load_list(self, root, *filenames):
        output = []
        for filename in filenames:
            filepath = os.path.join(root, filename)
            with open(filepath) as fileobj:
                output += [os.path.normpath(os.path.join(os.path.join(root, 'audio'), line.strip())) for line in fileobj]
        return output


    def __load_speechcommands_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str]:
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        label = label.split('\\')[1]
       
        #speaker, _ = os.path.splitext(filename)
        #speaker, _ = os.path.splitext(speaker)

        #speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        #utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = self.__load_wav_file(filepath)
        
        if self._transform is not None:
            waveform, sample_rate = self.__transform(waveform, sample_rate)

        return waveform, sample_rate, label #, speaker_id, utterance_number
    

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
            waveform = np.pad(waveform, pad_width=(int(left_pad), int(right_pad)), mode='edge')
        
        waveform = torch.tensor(waveform)
        waveform = torch.unsqueeze(waveform, 0)
        return waveform, sample_rate