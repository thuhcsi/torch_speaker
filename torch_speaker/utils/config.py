from .yacs import CfgNode

cfg = CfgNode(new_allowed=True)

def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)

if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(cfg, file=f)

