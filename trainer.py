import os
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.build_dataloader import get_dataloader


class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

        self.lr = config["lr"]
        if config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=20
        )

        self.epochs = config["epoch"]
        self.criterion = config["criterion"]

        self.path = config["csv_path"]
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.save_path = config["ckpt_path"]
        self.writer = SummaryWriter()
        self.global_step = 0
        self.eval_step = config["eval_step"]
        self.best_val_loss = 1e8
        self.best_val_acc = 0

    def get_dataloader(self) -> Tuple[DataLoader]:
        return get_dataloader(self.path, iid=True)

    def train(self):
        for epoch in tqdm(self.epochs, desc="epoch"):
            result = self._train_epoch(epoch)

        self.writer.close()
        return self.global_step, self.best_val_loss, self.best_val_acc

    def _train_epoch(self, epoch: int):
        train_loss = 0.0
        train_acc = 0.0

        self.model.train()
        for step, batch in tqdm(enumerate(self.train_loader, desc="steps")):
            img, label = map(lambda x: x.to(self.device), batch)

            output = self.model(img)
            pred = None  ##
            batch_acc = None  ##

            self.optimizer.zero_grad()
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            # train_acc += batch_acc.item()

            self.global_step += 1

            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    "global step: {:3}, train loss: {.3f}, train acc: {.3f}, lr: {.3f}".format(
                        self.global_step, loss.item(), batch_acc.item(), self.lr
                    )
                )

        train_loss /= step + 1
        train_acc /= step + 1

        val_loss, val_acc = self._valid_epoch(epoch)

        self.writer.add_scalars("loss", {"val": val_loss, "train": train_loss}, self.global_step)
        self.writer.add_scalars("accuracy", {"val": val_acc, "train": train_acc}, self.global_step)

        tqdm.write(
            "global step: {:3}, val loss: {.3f}, val acc: {.3f}".format(
                self.global_step, val_loss, val_acc
            )
        )

        if val_loss < self.best_val_loss:
            name = "./best_model.ckpt"
            torch.save(self.model.state_dict(), os.path.join(self.save_path, name))
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc

    def _valid_epoch(self, epoch: int) -> Tuple[float]:
        val_loss = 0.0
        val_acc = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(enumerate(self.val_loader), desc="val steps"):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                loss = self.criterion(output, label)
                batch_acc = None  ##

                val_loss += loss.item()
                val_acc += batch_acc.item()

        val_loss /= step + 1
        val_acc /= step + 1

        return val_loss, val_acc

