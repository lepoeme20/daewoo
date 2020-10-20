import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

from model import ResNet34
from trainer import Trainer


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    config = {
        "csv_path": args.path,
        "ckpt_path": args.ckpt_path,
        "epoch": args.epoch,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "criterion": nn.MSELoss(),
        "eval_step": args.eval_step,
        "fc_bias": bias,
    }

    fix_seed(args.seed)
    bias = True if args.bias else False
    model = ResNet34(num_classes=args.num_classes, fc_bias=bias)
    trainer = Trainer(model=model, config=config)

    t = time.time()
    global_step, best_val_loss = trainer.train()
    train_time = time.time() - t
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)

    print()
    print("Training Finished.")
    print("** Total Time: {}-hour {}-minute".format(int(h), int(m)))
    print("** Total Step: {}".format(global_step))
    print("** Best Validation Loss: {:.3f}".format(best_val_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument(
        "--path", type=str, default="./preprocessing/brave_data_label.csv"
    )
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=100)

    parser.add_argument(
        "--bias", type=str2bool, default="true", help="bias in fc layer"
    )

    args = parser.parse_args()
    main(args)
