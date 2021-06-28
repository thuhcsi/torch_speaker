# Copyright 2021 Yang Zhang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, coef=0.97):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop = hop
        self.pre_emphasis = PreEmphasis(coef)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_fft//2+1)

    def forward(self, x):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=torch.hamming_window(self.win_length),
                       win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = self.instance_norm(x)
        x = x.unsqueeze(1)
        return x


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, n_mels=64, coef=0.97, requires_grad=False):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

    def forward(self, x):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        x = self.instance_norm(x)
        x = x.unsqueeze(1)
        return x


class Multi_Resolution_Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=[1024, 1024, 1024, 1024, 1024], 
            win_length=[100, 200, 400, 800, 1024],
            hop=[160, 160, 160, 160, 160], n_mels=64, coef=0.97):
        super(Multi_Resolution_Mel_Spectrogram, self).__init__()
        assert len(n_fft) == len(win_length)
        assert len(n_fft) == len(hop)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)

        self.mel_trans = []
        for i in range(len(n_fft)):
            mel = Mel_Spectrogram(sample_rate, n_fft[i], win_length[i], hop[i], n_mels, coef)
            self.mel_trans.append(mel)
        self.mel_trans = nn.ModuleList(self.mel_trans)

    def forward(self, waveform):
        waveform = self.pre_emphasis(waveform)
        res = []
        n_frames = -1
        for i in range(len(self.n_fft)):
            fbank = self.mel_trans[i](waveform)
            if i == 0:
                n_frames = fbank.shape[-1]
            elif n_frames != fbank.shape[-1]:
                fbank = functional.resize(fbank, (self.n_mels, n_frames))
            res.append(fbank)
        res = torch.cat(res, axis=1)
        return res


if __name__ == "__main__":
    from scipy.io import wavfile
    import matplotlib.pyplot as plt

    sample_rate, sig = wavfile.read("test.wav")
    sig = torch.FloatTensor(sig.copy())
    sig = sig.repeat(10, 1)

    spec = Multi_Resolution_Mel_Spectrogram()
    out = spec(sig)
    out = out.detach().numpy()
    print(out.shape)
    out = out[0]
    plt.subplot(511)
    plt.imshow(out[0])
    plt.subplot(512)
    plt.imshow(out[1])
    plt.subplot(513)
    plt.imshow(out[2])
    plt.subplot(514)
    plt.imshow(out[3])
    plt.subplot(515)
    plt.imshow(out[4])
    plt.show()
    plt.clf()

    #plt.subplot(321)
    #plt.title("raw waveform")
    #plt.plot(sig[0])

    #plt.subplot(322)
    #plt.title("after PreEmphasis")
    #pre = PreEmphasis()
    #sig = pre(sig)
    #plt.plot(sig[0])

    #plt.subplot(323)
    #plt.title("Spectrogram($n_{fft}=512$)")
    #spec = Spectrogram()
    #out = spec(sig)
    #out = out.detach().numpy()
    #print(out.shape)
    #plt.imshow(out[0][0])

    #plt.subplot(324)
    #plt.title("Mel filter($n_{fft}=512, n_{mels}=64$)")
    #mel_basis = librosa.filters.mel(sr=16000, n_fft=512, n_mels=64)
    #plt.imshow(mel_basis)

    #plt.subplot(313)
    #plt.title("Mel-Spectrogram($n_{mels}=64$)")
    #mel = Mel_Spectrogram()
    #out = mel(sig)
    #out = out.detach().numpy()
    #plt.imshow(out[0][0])
    #out = out[0][0]
    #out[10:20] = 0

    #plt.tight_layout()
    #plt.show()
