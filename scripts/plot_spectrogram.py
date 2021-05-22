import scipy.signal as signal
from scipy.io import wavfile
import numpy as np
import librosa
import matplotlib.pyplot as plt


sample_rate, sig = wavfile.read("test.wav")
plt.subplot(211)
plt.plot(sig)

plt.subplot(212)
f, t, zxx = signal.stft(sig, fs=sample_rate, nperseg=200, noverlap=100, nfft=256)
zxx = np.log(np.abs(zxx))
T, F = np.meshgrid(t, f)
plt.contourf(T, F, zxx)
plt.show()
