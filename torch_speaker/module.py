import os
import copy

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_speaker.audio as audio
import torch_speaker.backbone as backbone
import torch_speaker.loss as loss
import torch_speaker.score as score

from torch_speaker.audio import Evaluation_Dataset, Train_Dataset
from torch_speaker.utils import count_spk_number


class Task(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset_name = self.hparams.train_dataset.pop('name')

        # 1. acoustic feature
        feature_name = self.hparams.feature.pop('name')
        self.feature = getattr(audio, feature_name)(**self.hparams.feature)

        # 2. backbone
        backbone_name = self.hparams.backbone.pop('name')
        self.backbone = getattr(backbone, backbone_name)(
            **self.hparams.backbone)

        # 3. compute loss function for gradient desent
        if os.path.exists(self.hparams.train_dataset.train_csv_path):
            self.hparams.loss.num_classes = count_spk_number(self.hparams.train_dataset.train_csv_path)
            loss_name = self.hparams.loss.pop('name')
            self.loss = getattr(loss, loss_name)(**self.hparams.loss)

    def extract_embedding(self, x):
        x = x.reshape(-1, x.shape[-1])
        x = self.feature(x)
        x = self.backbone(x)
        return x

    def forward(self, x, label=None):
        x = x.reshape(-1, x.shape[-1])
        x = self.feature(x)
        x = self.backbone(x)
        x = x.reshape(-1, self.hparams.num_shot, x.shape[-1])
        loss, acc = self.loss(x, label)
        return loss, acc

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        loss, acc = self(waveform, label)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def train_dataloader(self):
        print("init {} dataset".format(self.train_dataset_name))
        build_dataset = getattr(audio, self.train_dataset_name)
        dataset_cfg = copy.deepcopy(self.hparams.train_dataset)
        train_dataset = build_dataset(**dataset_cfg)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def on_test_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def on_validation_epoch_start(self):
        self.on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, path = batch
        path = str(path[0])
        x = self.extract_embedding(x)
        x = x.cpu().detach().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        labels, scores = score.cosine_score(
            self.trials, self.index_mapping, self.eval_vectors)
        EER, threshold = score.compute_eer(labels, scores)

        print("\ncosine EER: {:.2f}% with threshold {:.2f}".format(EER*100, threshold))
        self.log("cosine_eer", EER*100)

        minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.01)
        print("cosine minDCF(10-2): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-2)", minDCF)

        minDCF, threshold = score.compute_minDCF(labels, scores, p_target=0.001)
        print("cosine minDCF(10-3): {:.2f} with threshold {:.2f}".format(minDCF, threshold))
        self.log("cosine_minDCF(10-3)", minDCF)

        # save score
        if self.hparams.score_save_path is not None:
            with open(self.hparams.score_save_path, "w") as f:
                for i in range(len(labels)):
                    f.write("{} {}\n".format(labels[i], scores[i]))

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def test_dataloader(self):
        print("trial for evaluation: ", self.hparams.trial_path)
        trials = np.loadtxt(self.hparams.trial_path, str)
        self.trials = trials
        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(set(trials.T[1]))))
        print("number of test: {}".format(len(set(trials.T[2]))))
        print("number of evaluation: {}".format(len(eval_path)))
        eval_dataset = Evaluation_Dataset(eval_path)
        loader = torch.utils.data.DataLoader(eval_dataset,
                                             num_workers=self.hparams.num_workers,
                                             shuffle=False, batch_size=1)
        return loader

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        optim_name = self.hparams.optim.pop('name')
        build_optimizer = getattr(torch.optim, optim_name)
        optimizer_cfg = copy.deepcopy(self.hparams.optim)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)
        print("init {} optimizer with {}".format(
            optim_name, dict(optimizer_cfg)))

        schedule_cfg = copy.deepcopy(self.hparams.schedule)
        schedule_name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, schedule_name)
        lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        print("init {} lr_scheduler with {}".format(
            schedule_name, dict(schedule_cfg)))

        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.optim.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
