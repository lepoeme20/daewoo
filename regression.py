import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from model import ResNet34
from trainer import Trainer
from utils.build_dataloader import get_dataloader


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
    }

    fix_seed(args.seed)
    model = ResNet34(num_classes=args.num_classes)
    trainer = Trainer(model=model, config=config)

    t = time.time()
    global_step, best_val_loss = trainer.train()
    train_time = time.time() - t
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)

    print()
    print("Training Finished.")
    print("** Total Time: {}-hour {}-minute".format(h, m))
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
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=50)

    args = parser.parse_args()
    main(args)
