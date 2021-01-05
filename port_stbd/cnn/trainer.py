import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloaders
from utils import EarlyStopping


class Trainer:
    def __init__(self, config: dict, model: nn.Module):
        super().__init__()
        self.config = config

        # model hparam
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.root = config["root"]
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        self.num_classes = config["num_classes"]
        self.criterion = nn.CrossEntropyLoss()

        # training hparam
        self.epochs = config["epoch"]
        self.lr = config["lr"]
        self.optimizer = self.get_optimizer(config["optimizer"])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10
        )
        self.train_batch_size = config["train_batch_size"]
        self.eval_batch_size = config["eval_batch_size"]
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.root, self.train_batch_size, self.eval_batch_size
        )
        ## TODO: val, test loader

        # model saving hparam
        self.save_path = config["ckpt_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path)
        self.global_step = 0
        self.eval_step = config["eval_step"]
        self.earlystopping = EarlyStopping(
            verbose=True, path=os.path.join(self.save_path)
        )

    def get_optimizer(self, opt) -> optim:
        if opt == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        elif opt == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.lr)

    def train(self) -> None:

        for epoch in tqdm(range(self.epochs), desc="epoch"):
            val_loss = self._train_epoch(epoch)

            self.earlystopping(epoch, val_loss, self.model)
            if self.earlystopping.early_stop:
                print("Early Stopped.")
                break

        self.writer.close()

    def _train_epoch(self, epoch: int) -> float:
        train_loss = 0.0

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader), desc="steps", total=len(self.train_loader)
        ):
            img, label = map(lambda elm: elm.to(self.device), batch)
            output = self.model(img)

            self.optimizer.zero_grad()
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    "global step: {}, train loss: {:.3f}".format(
                        self.global_step, loss.item()
                    )
                )

        train_loss /= step + 1
        val_loss = self._valid_epoch(epoch)

        self.writer.add_scalars(
            "loss", {"val": val_loss, "train": train_loss}, self.global_step
        )
        tqdm.write(
            "** global step: {}, val loss: {:.3f}".format(self.global_step, val_loss)
        )
        self.lr_scheduler.step(val_loss)

        return val_loss

    def _valid_epoch(self, epoch: int) -> float:
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader), desc="val steps", total=len(self.val_loader)
            ):
                img, label = map(lambda elm: elm.to(self.device), batch)

                output = self.model(img)
                loss = self.criterion(output.squeeze(), label)

                val_loss += loss.item()

        val_loss /= step + 1

        return val_loss

    def test(self, best_model) -> Tuple[float]:
        test_loss = 0.0
        test_acc = 0.0
        correct = 0
        total = 0

        best_model = best_model.to(self.device)
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="test steps",
                total=len(self.test_loader),
            ):
                img, label = map(lambda elm: elm.to(self.device), batch)

                output = best_model(img)
                _, predicted = torch.max(output.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()
                loss = self.criterion(output.squeeze(), label)

                test_loss += loss.item()

        test_loss /= step + 1
        print("Accuracy of the network on test images: %d %%" % (100 * correct / total))
        print("Test loss of the network on test images: %.3f" % (test_loss))
        return test_loss
