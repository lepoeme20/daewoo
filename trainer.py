import logging
import os
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
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=20
        )

        self.epochs = config["epoch"]
        self.criterion = config["criterion"]

        self.path = config["csv_path"]
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()

        # self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.INFO)
        self.save_path = config["ckpt_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path)
        self.global_step = 0
        self.eval_step = config["eval_step"]
        self.best_val_loss = 1e8

    def get_dataloader(self) -> Tuple[DataLoader]:
        return get_dataloader(self.path, iid=False)

    def train(self) -> Tuple[int, float]:
        for epoch in tqdm(range(self.epochs), desc="epoch"):
            result = self._train_epoch(epoch)

        self.writer.close()
        return self.global_step, self.best_val_loss

    def _train_epoch(self, epoch: int) -> None:
        train_loss = 0.0

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader), desc="steps", total=len(self.train_loader)
        ):
            img, label = map(lambda x: x.to(self.device), batch)

            output = self.model(img)

            self.optimizer.zero_grad()
            loss = self.criterion(output.squeeze(), label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    "global step: {}, train loss: {:.3f}, lr: {:.6f}".format(
                        self.global_step, loss.item(), self.lr
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

        if val_loss < self.best_val_loss:
            name = "best_model.ckpt"
            torch.save(self.model.state_dict(), os.path.join(self.save_path, name))
            self.best_val_loss = val_loss

    def _valid_epoch(self, epoch: int) -> Tuple[float]:
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader), desc="val steps", total=len(self.val_loader)
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                loss = self.criterion(output, label)

                val_loss += loss.item()

        val_loss /= step + 1

        return val_loss
