import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from tqdm import tqdm
import copy
from collections import OrderedDict

import torch_speaker.audio as audio
from torch_speaker.audio import Train_Dataset, Evaluation_Dataset
import torch_speaker.backbone as backbone
import torch_speaker.loss as loss
import torch_speaker.score as score

class Task(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # 1. acoustic feature
        feature_name = self.hparams.feature.pop('name')
        self.feature = getattr(audio, feature_name)(**self.hparams.feature)

        # 2. backbone
        backbone_name = self.hparams.backbone.pop('name')
        self.backbone = getattr(backbone, backbone_name)(**self.hparams.backbone)

        # 3. compute loss function for gradient desent
        loss_name = self.hparams.loss.pop('name')
        self.loss = getattr(loss, loss_name)(**self.hparams.loss)

    def extract_embedding(self, x):
        x = self.feature(x)
        x = self.backbone(x)
        return x

    def forward(self, x, label):
        x = self.feature(x)
        x = self.backbone(x)
        x = x.unsqueeze(1)
        loss, acc = self.loss(x, label)
        return loss, acc

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        loss, acc = self(waveform, label)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def train_dataloader(self):
        train_dataset = Train_Dataset(self.hparams.train_csv, self.hparams.second)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                drop_last=False,
                )
        return loader

    def evaluation(self):
        trials = np.loadtxt(self.hparams.trial_path, str)
        labels = trials.T[0].astype(int)

        eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
        print("number of enroll: {}".format(len(trials.T[1])))
        print("number of test: {}".format(len(trials.T[2])))
        print("number of evaluation: {}".format(len(eval_path)))
        eval_dataset = Evaluation_Dataset(eval_path)
        loader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=1)
        index_mapping = {}
        eval_vectors = [ None for _ in range(len(loader))]
        for idx, x in enumerate(tqdm(loader)):
            x = x.cuda()
            x = self.extract_embedding(x)
            x = x.cpu().detach().numpy()[0]
            index_mapping[eval_path[idx]] = idx
            eval_vectors[idx] = x

        eer, th = score.cosine_score(trials, index_mapping, eval_vectors)
        print(eer)
 
    def configure_optimizers(self):
        optim_name = self.hparams.optim.pop('name')
        build_optimizer = getattr(torch.optim, optim_name)
        optimizer_cfg = copy.deepcopy(self.hparams.optim)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)
        print("init {} optimizer with {}".format(optim_name, dict(optimizer_cfg)))

        schedule_cfg = copy.deepcopy(self.hparams.schedule)
        schedule_name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, schedule_name)
        lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        print("init {} lr_scheduler with {}".format(schedule_name, dict(schedule_cfg)))

        return [optimizer], [lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.optim.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

