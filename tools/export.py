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
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file path', default=None)
    parser.add_argument('--onnx_save_path', type=str, help='onnx file path', default=None)
    parser.add_argument('--frame_length', type=int, help="frame length", default=250)
    args = parser.parse_args()
    load_config(cfg, args.config)

    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path

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

    feature = model.feature
    feature = feature.eval()
    backbone = model.backbone
    backbone = backbone.eval()

    x = torch.randn(10, 1, cfg.feature.n_mels, args.frame_length)
    torch.onnx._export(backbone, x, args.onnx_save_path, export_params=True)

