import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list_path', type=str, default="data/train.csv")
    parser.add_argument('--cohort_save_path', type=str, default="data/cohort.csv")
    parser.add_argument('--num_cohort', type=int, default=3000)
    args = parser.parse_args()

