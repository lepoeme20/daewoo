import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from model import ResNet
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
        "criterion": nn.CrossEntropyLoss(),
        "eval_step": args.eval_step,
    }

    fix_seed(args.seed)
    model = ResNet()  # TODO: num_classes 추가
    trainer = Trainer(model=model, config=config)

    t = time.time()
    global_step, best_val_loss, best_val_acc = trainer.train()
    train_time = time.time() - t

    print()
    print("Training Finished.")
    print("** Time: {.3f}".format(train_time))
    print("** Total Step: {}".format(global_step))
    print("** Best Validation Loss: {.3f}".format(best_val_loss))
    print("** Best Validation Accuracy: {.3f}%".format(best_val_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--path", type=str, default="./brave_data_label.csv")  ##
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--eval_step", type=int, default=20)

    args = parser.parse_args()
    main(args)
