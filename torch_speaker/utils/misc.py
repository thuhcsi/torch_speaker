import os
import numpy as np
import pandas as pd

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_spk_number(train_csv_path):
    df = pd.read_csv(train_csv_path)
    data_labels = df["utt_spk_int_labels"].values
    return len(set(data_labels))

def compute_dB(waveform):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    dB = 10*np.log10(val+1e-4)
    return dB

def compute_SNR(waveform, noise):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    SNR = 10*np.log10(np.mean(waveform**2)/np.mean(noise**2)+1e-9)
    return SNR


