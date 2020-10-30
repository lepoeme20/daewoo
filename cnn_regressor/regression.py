import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from model import ResNet34
from trainer import Trainer
from model_utils import str2bool, fix_seed


def main(args):
    bias = True if args.bias else False
    config = {
        "csv_path": args.path,
        "ckpt_path": args.ckpt_path,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "criterion": nn.MSELoss(),
        "eval_step": args.eval_step,
        "fc_bias": bias,
        "iid": args.iid,
        "transform": args.transform,
    }

    fix_seed(args.seed)
    model = ResNet34(num_classes=args.num_classes, fc_bias=bias)
    trainer = Trainer(model=model, config=config)

    t = time.time()
    # global_step, best_val_loss = trainer.train()
    trainer.train()
    train_time = time.time() - t
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)

    print()
    print("Training Finished.")
    print("** Total Time: {}-hour {}-minute".format(int(h), int(m)))
    # print("** Total Step: {}".format(global_step))
    # print("** Best Validation Loss: {:.3f}".format(best_val_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument(
        "--path", type=str, default="../preprocessing/brave_data_label.csv"
    )
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=200)

    parser.add_argument(
        "--transform",
        type=int,
        default=0,
        help="0:no-norm / 1:whole-img-norm / 2:per-img-norm",
    )
    parser.add_argument("--iid", type=str2bool, default="false")
    parser.add_argument(
        "--bias", type=str2bool, default="false", help="bias in fc layer"
    )

    args = parser.parse_args()
    main(args)
