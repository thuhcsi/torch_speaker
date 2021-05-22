#!/usr/bin/env python
# coding=utf-8

import scipy.signal as signal

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

sample_rate, sig = wavfile.read("test.wav")

f, t, zxx = signal.stft(sig, fs=sample_rate, nperseg=400, noverlap=300, nfft=512)
zxx = np.log(np.abs(zxx))
T, F = np.meshgrid(t, f)

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

surf = ax.plot_surface(T, F, zxx, rstride=1, cstride=1, cmap=cm.viridis)
ax.contourf(T, F, zxx, zdir='z', offset=-30)
ax.set_zlim(-30, 10)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
