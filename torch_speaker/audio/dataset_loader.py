import random
import torch
import pandas as pd
import random
import os
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_audio(filename, second=2):
    sample_rate, waveform  = wavfile.read(filename)
    if second <= 0:
        return waveform

    length = sample_rate * second
    audio_length = waveform.shape[0]

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        return waveform
    else:
        start = np.int64(random.random()*(audio_length-length))
        return waveform[start:start+length]

class Train_Dataset(Dataset):
    def __init__(self, data_list_path, second, **kwargs):
        self.data_list_path = data_list_path
        self.second = second
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_path = df["utt_paths"].values
        print("Train Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Train Dataset load {} utterance".format(len(self.data_label)))

    def __getitem__(self, index):
        audio = load_audio(self.data_path[index], self.second)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_label)


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        audio = load_audio(self.paths[index], self.second)
        return torch.FloatTensor(audio)

    def __len__(self):
        return len(self.paths)

