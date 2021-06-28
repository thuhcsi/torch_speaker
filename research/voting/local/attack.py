import argparse
import torch

from torch_speaker.utils import cfg, load_config
from torch_speaker.module import Task

class Adversarial_Attack_Helper(object):
    def __init__(self, model, alpha=3.0, restarts=1, num_iters=5, epsilon=15, 
            adv_save_dir="data/adv_data/", device="cuda"):
        self.model = model
        self.alpha = alpha
        self.restarts = restarts
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.adv_save_dir = adv_save_dir
        self.adv_trials_path = os.path.join(adv_save_dir, "adv_trials.lst")
        self.device = device

        if not os.path.exists(os.path.join(adv_save_dir, "wav")):
            os.makedirs(os.path.join(adv_save_dir, "wav"))

        self.model.eval()
        if self.device == "cuda":
            self.model.cuda()

    def attack(self):
        # adversarial attack example generation
        adv_trials_file = open(self.adv_trials_path, "w")
        labels = []
        scores = []
        for idx, item in enumerate(tqdm(self.trials)):
            pass

        print("EER: {:.3f} %".format(eer*100))


    def bim_attack_step(self, idx, item):
        label, enroll_path, test_path = item
        return label, enroll_path, adv_test_path, final_score



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

