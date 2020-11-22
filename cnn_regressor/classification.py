import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from model import ResNet34
from trainer import Trainer
from model_utils import str2bool, fix_seed, softXEnt


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
        "criterion": softXEnt,
        "eval_step": args.eval_step,
        "fc_bias": bias,
        "label_type": args.label_type,
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
        "--path", type=str, default="./preprocessing/brave_data_label.csv"
    )
    parser.add_argument("--ckpt-path", type=str, default="./checkpoints/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval-step", type=int, default=200)

    parser.add_argument(
        "--transform",
        type=int,
        default=2,
        help="0:no-norm / 1:whole-img-norm / 2:per-img-norm",
    )
    parser.add_argument("--label-type", type=int, default=1)
    parser.add_argument("--iid", type=str2bool, default="true")
    parser.add_argument(
        "--bias", type=str2bool, default="true", help="bias in fc layer"
    )

    args, _ = parser.parse_known_args()

    if args.label_type == 0:
        print(" Set label as height")
        args.label_type = 'height'
    elif args.label_type == 1:
        print(" Set label as direction")
        args.label_type = 'direction'
    elif args.label_type == 2:
        print(" Set label as period")
        args.label_type = 'period'

    main(args)
