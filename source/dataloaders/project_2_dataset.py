import os
from pathlib import Path
from typing import Tuple, Optional, Union
from torch.utils.data import Dataset

import torch
import torchaudio
from torch import Tensor

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"

class Project_2_Dataset(Dataset):

    def __init__(
            self,
            root: Union[str, Path],
            subset: Optional[str] = None,
            transform: torch.nn.Sequential = None
        ) -> None:

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from " + "{'training', 'validation', 'testing'}."
        )

         # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._path = root
        self._transform = transform
        

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
                if os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob("*/*.wav"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]

        
    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, int):
            ``(waveform, sample_rate, label, speaker_id, utterance_number)``
        """
        fileid = self._walker[n]
        return self.__load_speechcommands_item(fileid, self._path)


    def __len__(self) -> int:
        return len(self._walker)

    def __load_list(self, root, *filenames):
        output = []
        for filename in filenames:
            filepath = os.path.join(root, filename)
            with open(filepath) as fileobj:
                output += [os.path.normpath(os.path.join(os.path.join(root, 'audio'), line.strip())) for line in fileobj]
        return output

    def __load_speechcommands_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
       
        speaker, _ = os.path.splitext(filename)
        speaker, _ = os.path.splitext(speaker)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = torchaudio.load(filepath)

        if self._transform is not None:
             waveform, sample_rate = self.__transform(waveform, sample_rate)

        return waveform, sample_rate, label, speaker_id, utterance_number
    

    def __transform(self, waveform: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        pass