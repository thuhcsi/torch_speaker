import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset dir', type=str, default="CN_Celeb")
    parser.add_argument('--output_trial_path', help='output trial path', type=str, default="data/trails.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    enroll_lst_path = os.path.join(args.dataset_dir, "eval/lists/enroll.lst")
    raw_trial_path = os.path.join(args.dataset_dir, "eval/lists/trials.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in enroll_lst:
        spk2wav_mapping[item[0]] = item[1]
    trials = np.loadtxt(raw_trial_path, str)

    with open(args.output_trial_path, "w") as f:
        for item in trials:
            enroll_path = os.path.join(args.dataset_dir, "eval", spk2wav_mapping[item[0]])
            test_path = os.path.join(args.dataset_dir,"eval", item[1])
            if args.apply_vad:
                enroll_path = enroll_path.strip(".wav") + ".vad"
                test_path = test_path.strip(".wav") + ".vad"
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))
