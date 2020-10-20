import argparse
import random
from collections import OrderedDict

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
    bias = True if args.bias else False
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
    model = ResNet34(num_classes=args.num_classes, fc_bias=bias)

    checkpoint = (
        torch.load("checkpoints/best_model_bias_true.ckpt")
        if bias
        else torch.load("checkpoints/best_model_bias_false.ckpt")
    )
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    trainer = Trainer(model=model, config=config)

    if args.mode == "test":
        mse, mae, mape = trainer.test()
        print()
        print("Test Finished.")
        print("** Test Loss (MSE): {:.3f}".format(mse))
        print("** Test Loss (MAE): {:.3f}".format(mae))
        print("** Test Loss (MAPE): {:.3f}".format(mape))
        return

    pred, true = trainer.test_values()
    return pred, true


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument(
        "--path", type=str, default="./preprocessing/brave_data_label.csv"
    )
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=50)

    parser.add_argument(
        "--bias", type=str2bool, default="true", help="bias in fc layer"
    )
    parser.add_argument("--mode", type=str, default="test")

    args = parser.parse_args()
    main(args)
