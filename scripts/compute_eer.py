import argparse
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def compute_eer(labels, scores):
    """sklearn style compute eer
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_path', type=str, default="score.txt")
    parser.add_argument('--score_img', type=str, default=None)
    args = parser.parse_args()

    data = np.loadtxt(args.score_path).T
    labels = data[0]
    scores = data[1]
    eer, threshold = compute_eer(labels, scores)
    print("EER: {:.3f}% with threshold {:.2f}".format(eer*100, threshold))

    if args.score_img is not None:
        false_score = []
        true_score = []
        for i in range(len(labels)):
            if labels[i] == 0:
                false_score.append(scores[i])
            else:
                true_score.append(scores[i])

        plt.hist(false_score, bins=100, label="negtive score")
        plt.hist(true_score, bins=100, label="positive score")

        plt.legend()
        plt.tight_layout()
        plt.savefig(args.score_img)
