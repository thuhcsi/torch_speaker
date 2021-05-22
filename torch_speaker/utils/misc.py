import os
import pandas as pd

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_spk_number(train_csv_path):
	df = pd.read_csv(train_csv_path)
	data_labels = df["utt_spk_int_labels"].values
	return len(set(data_labels))

