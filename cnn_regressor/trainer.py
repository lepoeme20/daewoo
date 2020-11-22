import logging
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader
from model_utils import EarlyStopping


class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        ## model param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bias = config["fc_bias"]
        self.label_type = config["label_type"]
        self.iid = config["iid"]
        self.transform = config["transform"]
        self.path = config["csv_path"]
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

        ## training param
        self.epochs = config["epoch"]
        self.criterion = config["criterion"]
        self.batch_size = config["batch_size"]
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
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()

        ## model saving param
        self.save_path = config["ckpt_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path)
        self.global_step = 0
        self.eval_step = config["eval_step"]
        # self.best_val_loss = 1e8
        self.earlystopping = EarlyStopping(
            verbose=True, path=os.path.join(self.save_path, "best_model.ckpt")
        )

    def get_dataloader(self) -> Tuple[DataLoader]:
        return get_dataloader(
            self.path,
            batch_size=self.batch_size,
            label_type=self.label_type,
            iid=self.iid,
            transform=self.transform,
        )

    def train(self) -> None:  # -> Tuple[int, float]:
        for epoch in tqdm(range(self.epochs), desc="epoch"):
            val_loss = self._train_epoch(epoch)
            self.earlystopping(val_loss, self.model)
            if self.earlystopping.early_stop:
                print("Early Stopped.")
                break

        self.writer.close()
        # return self.global_step, self.best_val_loss

    def _train_epoch(self, epoch: int) -> float:
        train_loss = 0.0
        start_time = time.time()

        self.model.train()
        for step, (img, label) in tqdm(
            enumerate(self.train_loader), desc="steps", total=len(self.train_loader)
        ):
            img = img.to(self.device)
            if self.label_type != "direction":
                label = label.to(self.device)
            output = self.model(img)

            self.optimizer.zero_grad()
            if self.label_type == 'direction':
                loss = self.criterion(output, label, window_size=2)
            else:
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

        # if val_loss < self.best_val_loss:
        #     name = f"best_model_bias_{self.bias}.ckpt"
        #     torch.save(self.model.state_dict(), os.path.join(self.save_path, name))
        #     self.best_val_loss = val_loss

        self.lr_scheduler.step(val_loss)

        elapsed = time.time() - start_time
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        tqdm.write(
            "*** Epoch {} ends, it takes {}-hour {}-minute".format(
                epoch, int(h), int(m)
            )
        )

        return val_loss

    def _valid_epoch(self, epoch: int) -> float:
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, (img, label) in tqdm(
                enumerate(self.val_loader), desc="val steps", total=len(self.val_loader)
            ):
                img = img.to(self.device)
                if self.label_type != "direction":
                    label = label.to(self.device)

                output = self.model(img)
                if self.label_type == 'direction':
                    loss = self.criterion(output, label, window_size=2)
                else:
                    loss = self.criterion(output.squeeze(), label)

                val_loss += loss.item()

        val_loss /= step + 1

        return val_loss

    def test(self) -> Tuple[float]:
        loss_mse = 0.0
        loss_mae = 0.0
        loss_mape = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="test steps",
                total=len(self.test_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                mseloss = self.criterion(output.squeeze(), label)
                maeloss = self.MAE(output.squeeze(), label)
                mapeloss = self.MAPE(output.squeeze(), label)

                loss_mse += mseloss.item()
                loss_mae += maeloss.item()
                loss_mape += mapeloss.item()

        loss_mse /= step + 1
        loss_mae /= step + 1
        loss_mape /= step + 1

        return loss_mse, loss_mae, loss_mape

    def test_values(self, length: int = 50):
        pred = []
        true = []

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="test values",
                total=len(self.test_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                output = output.squeeze()
                pred.append(output)
                true.append(label)

                if step > length:
                    break

        return pred, true

    def MAE(self, pred: List[torch.tensor], true: List[torch.tensor]) -> torch.tensor:
        return torch.mean((pred - true).abs())

    def MAPE(self, pred: List[torch.tensor], true: List[torch.tensor]) -> torch.tensor:
        return torch.mean((pred - true).abs() / (true.abs() + 1e-8)) * 100  # percentage