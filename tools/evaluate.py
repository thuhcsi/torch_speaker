import argparse
import torch

from torch_speaker.utils import cfg, load_config
from torch_speaker.module import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='train config file path', default="config/config.yaml")
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path', default=None)
    parser.add_argument('--trial_path', type=str, help='trial file path', default=None)

    args = parser.parse_args()
    load_config(cfg, args.config)
    cfg.trainer.gpus = 1
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
    trainer = Trainer(default_root_dir=cfg.workspace, **cfg.trainer)
    trainer.test(model)
