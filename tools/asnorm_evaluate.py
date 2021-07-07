import numpy as np
import argparse
import torch
from tqdm import tqdm

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

from torch_speaker.utils import cfg, load_config
from torch_speaker.module import Task
from pytorch_lightning import Trainer
from torch_speaker.audio import Evaluation_Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback


def compute_eer(labels, scores):
    """sklearn style compute eer
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', default="config/config.yaml")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path', default=None)
    parser.add_argument('--trial_path', type=str, help='trial file path', default=None)
    parser.add_argument('--cohort_path', type=str, 
            help='a cohort with N speakers which is different from the enroll and tes', default=None)
    parser.add_argument('--score_save_path', type=str, help='trial file path', default=None)
    parser.add_argument('--top_n', type=int, help='trial file path', default=300)

    args = parser.parse_args()
    load_config(cfg, args.config)
    cfg.trainer.gpus = 1
    cfg.score_save_path = args.score_save_path
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    if args.trial_path is not None:
        cfg.trial_path = args.trial_path

    model = Task(**cfg)
    if cfg.checkpoint_path is not None:
        state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")["state_dict"]
        # pop loss Function parameter
        loss_weights = []
        if cfg.keep_loss_weight is False:
            for key, value in state_dict.items():
                if "loss" in key:
                    loss_weights.append(key)
            for item in loss_weights:
                state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(cfg.checkpoint_path))
        print("keep_loss_weight {}".format(cfg.keep_loss_weight))

    model = model.eval().cuda()
    cohort = np.loadtxt(args.cohort_path, str)
    cohort_dataset = Evaluation_Dataset(cohort)
    loader = torch.utils.data.DataLoader(cohort_dataset,
                                         num_workers=10,
                                         shuffle=False, batch_size=1)
    cohort_embeddings = []
    with torch.no_grad():
        for audio, audio_label in tqdm(loader):
            audio = audio.cuda()
            embedding = model.extract_embedding(audio).detach().cpu().numpy()[0]
            cohort_embeddings.append(embedding)
    cohort_embeddings = np.array(cohort_embeddings)

    trials = np.loadtxt(args.trial_path, str)
    eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    print("number of enroll: {}".format(len(set(trials.T[1]))))
    print("number of test: {}".format(len(set(trials.T[2]))))
    print("number of evaluation: {}".format(len(eval_path)))
    eval_dataset = Evaluation_Dataset(eval_path)
    loader = torch.utils.data.DataLoader(eval_dataset,
                                         num_workers=10,
                                         shuffle=False, batch_size=1)
    #table:
    #   key: audio_label
    #   val: [embedding, mean, std]
    table = {}
    with torch.no_grad():
        for audio, audio_label in tqdm(loader):
            val = []
            audio = audio.cuda()
            embedding = model.extract_embedding(audio).detach().cpu().numpy()[0]

            scores = embedding.dot(cohort_embeddings.T)
            denoms = np.linalg.norm(embedding) * np.linalg.norm(cohort_embeddings, axis=1)
            scores = scores / denoms
            scores = np.sort(scores)
            val.append(embedding)
            val.append(np.mean(scores[-args.top_n:]))
            val.append(np.std(scores[-args.top_n:]))
            table[audio_label[0]] = val

    scores = []
    labels = []
    for label, enroll_path, test_path in tqdm(trials):
        enroll_vector = table[enroll_path][0]
        enroll_mean = table[enroll_path][1]
        enroll_std = table[enroll_path][2]

        test_vector = table[test_path][0]
        test_mean = table[test_path][1]
        test_std = table[test_path][2]

        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom

        score = 0.5 * (score-enroll_mean) / enroll_std + \
                0.5 * (score-test_mean) / test_std

        scores.append(score)
        labels.append(int(label))

    eer, threshold = compute_eer(labels, scores)
    print("EER: {:.3f}% with threshold {:.2f}".format(eer*100, threshold))

    with open(args.score_save_path, "w") as f:
        for label, score in zip(labels, scores):
            f.write("{} {}\n".format(label, score))

