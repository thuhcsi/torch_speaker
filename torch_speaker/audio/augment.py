import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
from torch_speaker.utils import compute_dB, compute_SNR

class WavAugment(object):
    def __init__(self, noise_paths, noise_snr=[25, 30], guassian_snr=[-10, -1], volum=[0.6, 1.2]):
        self.noise_paths = noise_paths
        self.noise_snr = noise_snr
        self.guassian_snr = guassian_snr
        self.volum = volum

    def add_gaussian_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        snr = np.random.randint(self.guassian_snr[0], self.guassian_snr[1])
        clean_dB = compute_dB(waveform)
        noise = np.random.randn(len(waveform))
        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform.astype(np.int16)

    def change_volum(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        volum = np.random.uniform(low=self.volum[0], high=self.volum[1])
        waveform = waveform * volum
        return waveform.astype(np.int16)

    def change_speed(self, waveform):
        """
        Args:
            x (numpy.array): Input waveform array (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        pass

    def add_real_noise(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        snr = np.random.randint(self.noise_snr[0], self.noise_snr[1])
        clean_dB = compute_dB(waveform)

        idx = np.random.randint(0, len(self.noise_paths)-1)
        sample_rate, noise = wavfile.read(self.noise_paths[idx])
        noise_length = len(noise)
        audio_length = waveform.shape[0]

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            start = np.random.randint(0, (noise_length-audio_length))
            noise = noise[start:start+audio_length]

        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform.astype(np.int16)


class SpecAugment(nn.Module):
    def __init__(self, aug_ratio=1, freq_range=[1, 5], time_range=[1, 15]):
        super(SpecAugment, self).__init__()
        self.aug_ratio = aug_ratio
        self.freq_range = freq_range
        self.time_range = time_range

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, n_mels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels, n_mels, frames).
        """
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x

    def time_mask(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, n_mels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels, n_mels, frames).
        """
        batch_size = x.shape[0]
        aug_size = int(batch_size*self.aug_ratio)
        if aug_size == 0:
            return x

        index = [ i for i in range(batch_size)]
        np.random.shuffle(index)
        index = index[:aug_size]

        mask_len = np.random.randint(self.time_range[0], self.time_range[1])
        start = np.random.randint(0, x.shape[-1]-mask_len-1)
        x[index,:,:,start:start+mask_len] = 0
        return x

    def freq_mask(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, n_mels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels, n_mels, frames).
        """
        batch_size = x.shape[0]
        aug_size = int(batch_size*self.aug_ratio)
        if aug_size == 0:
            return x

        index = [i for i in range(batch_size)]
        np.random.shuffle(index)
        index = index[:aug_size]

        mask_len = np.random.randint(self.freq_range[0], self.freq_range[1])
        start = np.random.randint(0, x.shape[-2]-mask_len-1)
        x[index,:,start:start+mask_len,:] = 0
        return x


if __name__ == "__main__":
	import pandas as pd
	df = pd.read_csv("noise.csv")
	noise_paths = df["utt_paths"].values

	sample_rate, waveform = wavfile.read("test.wav")
	wave_aug = WavAugment(noise_paths=noise_paths)
	waveform = wave_aug.add_real_noise(waveform)
	waveform = wave_aug.add_gaussian_noise(waveform)
	waveform = wave_aug.change_volum(waveform)
	wavfile.write("out.wav", 16000, waveform.astype(np.int16))

	import matplotlib.pyplot as plt
	data = torch.randn(20, 1, 64, 200)

	plt.subplot(211)
	plt.imshow(data[0][0])

	spec_aug = SpecAugment(aug_ratio=1.0)
	data = spec_aug(data)
	plt.subplot(212)
	plt.imshow(data[0][0])
	plt.savefig("test.png")

