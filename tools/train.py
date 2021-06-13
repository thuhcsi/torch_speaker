import argparse
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)

from torch_speaker.module import Task
from torch_speaker.utils import cfg, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path',
                        default="config/config.yaml")
    args = parser.parse_args()
    load_config(cfg, args.config)

    model = Task(**cfg)
    if cfg.checkpoint_path is not None:
        state_dict = torch.load(cfg.checkpoint_path, map_location="cpu")[
            "state_dict"]
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

    checkpoint_callback = ModelCheckpoint(monitor='cosine_eer', save_top_k=cfg.save_top_k,
                                          filename="{epoch}_{cosine_eer:.2f}", dirpath=cfg.workspace)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(callbacks=[checkpoint_callback, lr_monitor],
					  num_sanity_val_steps=-1,
					  reload_dataloaders_every_epoch=True,
                      default_root_dir=cfg.workspace, **cfg.trainer)
    trainer.fit(model)
