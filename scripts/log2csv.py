from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--input_log_path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--csv_out_path', type=str, required=True, help='location to save the exported data')

    args = parser.parse_args()
    event_data = event_accumulator.EventAccumulator(args.in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so fa b
    print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    keys = keys[1:]
    print(keys)
    df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys):
        print(key)
        val = pd.DataFrame(event_data.Scalars(key))["value"]
        df[key] = val

    df.to_csv(args.ex_path)
    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    main()
