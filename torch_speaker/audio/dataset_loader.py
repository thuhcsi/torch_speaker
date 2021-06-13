import collections
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset

from .augment import WavAugment

def load_audio(filename, second=2):
    sample_rate, waveform = wavfile.read(filename)
    if second <= 0:
        return waveform

    length = np.int64(sample_rate * second)
    audio_length = waveform.shape[0]

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        return waveform
    else:
        start = np.int64(random.random()*(audio_length-length))
        return waveform[start:start+length].copy()

class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, noise_csv_path, second, spk_utt=200, num_per_speaker=1, **kwargs):
        self.second = second

        df = pd.read_csv(train_csv_path)
        data_labels = df["utt_spk_int_labels"].values
        data_paths = df["utt_paths"].values
        data_labels, data_paths = shuffle(data_labels, data_paths)

        df = pd.read_csv(noise_csv_path)
        noise_paths = df["utt_paths"].values
        self.wav_aug = WavAugment(noise_paths)

        table = {}
        for idx, label in enumerate(data_labels):
            if label not in table:
                table[label] = []
            table[label].append(data_paths[idx])

        self.labels = []
        self.paths = []
        for _ in range(spk_utt//num_per_speaker):
            for key, val in table.items():
                for _ in range(num_per_speaker):
                    idx = random.randint(0, len(val)-1)
                    self.labels.append(key)
                    self.paths.append(val[idx])
        print("Train Dataset load {} speakers".format(len(set(data_labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform = load_audio(self.paths[index], self.second)
		#aug_idx = np.random.randint(0, 3)
        #if aug_idx == 1:
        #    waveform = self.wav_aug.change_volum(waveform)
        #elif aug_idx == 2:
        #    waveform = self.wav_aug.add_gaussian_noise(waveform)
        #elif aug_idx == 3:
        #    waveform = self.wav_aug.add_real_noise(waveform)
        return torch.FloatTensor(waveform), self.labels[index]

    def __len__(self):
        return len(self.paths)


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        waveform = load_audio(self.paths[index], self.second)
        return torch.FloatTensor(waveform), self.paths[index]

    def __len__(self):
        return len(self.paths)
