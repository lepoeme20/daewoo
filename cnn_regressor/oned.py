import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader

# construct 1d CNN model

class CNN(nn.Module):
    def __init__(self, kernel_num, kernel_size, stride):
        super().__init__()
        self.conv_h = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, 1344), stride=stride) for k in kernel_size]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, 1)

    def forward(self, x):
        # set dim: (batch_size, max_seq_len, embedding_size) -> (batch_size, 1, max_seq_len, embedding_size)

        h = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.conv_h]

        # dim: [(batch_size, num_kernels), ...] * len(kernel_size)
        h = [nn.functional.max_pool1d(i, i.size(2)) for i in h]
        h = torch.cat(h, 1).squeeze(2)

        x = self.dropout(h)
        x = self.fc1(x)
        # x = self.fc2(x)

        return x


class Trainer:
    def __init__(self, args):
        self.device = args.device
        self.epochs = args.epochs
        self.stride = args.stride
        self.trn_data = args.trn_dataset
        self.tst_data = args.tst_dataset
        self.model = CNN(100, [20, 30, 40], self.stride)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr#, momentum=0.9, weight_decay=1e-3
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10
        )
        # data loader
        self.trn_loader, self.dev_loader, _ = get_dataloader(
            csv_path=args.trn_csv_path,
            batch_size=args.batch_size,
            label_type=args.label_type,
            iid=args.iid,
            transform=args.norm_type,
        )

        # set path
        self.model_path = (
            f"./cnn_regressor/best_model/oned/{args.trn_dataset}/{args.label_type}/norm_{args.norm_type}/{args.data_type}/seed_{args.iid}"
        )
        os.makedirs(self.model_path, exist_ok=True)

    def regression(self):
        # model
        self.model.to(self.device)
        # initial dev loss
        best_loss = 1000.0
        # model path
        model_path = os.path.join(self.model_path, f"{self.trn_data}_{self.stride}.pt")
        # criterion
        criterion = nn.MSELoss()

        outer = tqdm(total=args.epochs, desc="Epoch", position=0, leave=False)
        for epoch in range(self.epochs):
            self._train_loop(epoch, criterion)
            best_loss = self._dev_loop(epoch, criterion, best_loss, model_path)
            outer.update(1)

    def _train_loop(self, epoch, criterion):
        total_step = 0
        trn_loss_log = tqdm(total=0, position=2, bar_format="{desc}")
        train = tqdm(total=len(self.trn_loader), desc="Steps", position=1, leave=False)
        for step, (inputs, labels) in enumerate(self.trn_loader):
            self.model.train()
            total_step += 1

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            output = self.model(inputs)
            loss = criterion(torch.squeeze(output), labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            #################### Logging ###################
            trn_loss_log.set_description_str(
                f"[Train] Epoch: {epoch+1}/{args.epochs} Batch: {step+1}/{len(self.trn_loader)} Loss: {loss.item():.4f}"
            )
            train.update(1)

    def _dev_loop(self, epoch, criterion, best_loss, save_path):
        _dev_loss, dev_loss = 0.0, 0.0
        dev_loss_log = tqdm(total=0, position=4, bar_format="{desc}")
        best_epoch_log = tqdm(total=0, position=5, bar_format="{desc}")
        dev = tqdm(total=len(self.dev_loader), desc="Steps", position=3, leave=False)
        for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
            self.model.eval()
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                # Cross Entropy loss
                output = self.model(inputs)
                loss = criterion(torch.squeeze(output), labels)

                # Loss
                _dev_loss += loss
                dev_loss = _dev_loss / (idx + 1)

                dev_loss_log.set_description_str(
                    f"[Dev] {idx+1}/{len(self.dev_loader)} Loss: {dev_loss:.4f}"
                )
                dev.update(1)

        self.scheduler.step(dev_loss)

        if dev_loss < best_loss:
            best_epoch_log.set_description_str(f"Epoch: {epoch}, Loss: {dev_loss:.4f}")
            best_loss = dev_loss
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "trained_epoch": epoch,
                },
                save_path,
            )
        return best_loss

    def inference(self):
        # data loader
        _, _, tst_loader = get_dataloader(
            csv_path=args.tst_csv_path,
            batch_size=args.batch_size,
            label_type=args.label_type,
            iid=args.iid,
            transform=args.norm_type,
        )

        self.model.to(self.device)
        loss_mae = 0.0
        loss_mape = 0.0

        # load the saved model
        model_path = os.path.join(self.model_path, f"{self.trn_data}_{self.stride}.pt")
        checkpoint = torch.load(model_path)
        # check DataParallel
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(tst_loader),
                desc="test steps",
                total=len(tst_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)

                maeloss = self.MAE(torch.tensor(output, device=img.device), label)
                mapeloss = self.MAPE(torch.tensor(output, device=img.device), label)

                loss_mae += maeloss.item()
                loss_mape += mapeloss.item()

        loss_mae /= step + 1
        loss_mape /= step + 1

        return loss_mae, loss_mape

    def baseline(self):
        with torch.no_grad():
            trn_mean = torch.zeros((len(self.trn_loader), 1), device=self.device)
            for idx, (_, label) in tqdm(
                enumerate(self.trn_loader),
                desc="compute train mean",
                total=len(self.trn_loader),
            ):
                batch_height = label.to(self.device)
                trn_mean[idx] = torch.mean(batch_height).item()

            trn_mean = torch.mean(trn_mean)
            mae = 0.0
            mape = 0.0
            for step, (_, label) in tqdm(
                enumerate(self.tst_loader), desc="test step", total=len(self.tst_loader)
            ):
                y_tst = label.to(self.device)
                mean_value = torch.full_like(y_tst, trn_mean)
                mae += self.MAE(mean_value, y_tst).item()
                mape += self.MAPE(mean_value, y_tst).item()

            mae /= step + 1
            mape /= step + 1

        return mae, mape

    def MAE(self, pred, true):
        return torch.mean((pred - true).abs())

    def MAPE(self, pred, true):
        return torch.mean((pred - true).abs() / (true.abs() + 1e-8)) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn-dataset", type=str, default="weather", help="csv file path")
    parser.add_argument("--tst-dataset", type=str, default="weather", help="csv file path")
    parser.add_argument(
        "--root-csv-path",
        type=str,
        help="root directory of label csv file",
        default="./preprocessing/",
    )
    parser.add_argument(
        "--root-img-path",
        type=str,
        help="root directory of image folder",
        default="/hdd/data/daewoo/kisang1/",
    )
    parser.add_argument(
        "--norm-type",
        type=int,
        choices=[0, 1, 2],
        help="0: ToTensor, 1: Ordinary image normalizaeion, 2: Image by Image normalization",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Batch size")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00005, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Batch size")
    parser.add_argument("--seed", type=int, default=22, help="seed number")
    parser.add_argument(
        "--label-type",
        type=int,
        default=0,
        help="0: Height, 1: Direction, 2: Period, 3: Classification label",
    )
    parser.add_argument(
        "--iid",
        default=None,
        help="use argument for iid condition",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Perform Test only"
    )
    parser.add_argument(
        "--baseline", action="store_true", default=False, help="Train pretrained model"
    )
    parser.add_argument(
        "--stride", type=int, choices=[1, 5, 10]
    )

    args = parser.parse_args()

    args.trn_csv_path = os.path.join(args.root_csv_path, f"{args.trn_dataset}_data_label.csv")
    args.tst_csv_path = os.path.join(args.root_csv_path, f"{args.tst_dataset}_data_label.csv")
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.data_type = "iid" if args.iid is not None else "time"

    if args.label_type == 0:
        print(" Set label as height")
        args.label_type = "height"
    elif args.label_type == 1:
        print(" Set label as direction")
        args.label_type = "direction"
    elif args.label_type == 2:
        print(" Set label as period")
        args.label_type = "period"

    # Set random seed for reproducibility
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    trainer = Trainer(args)
    if args.baseline:
        mae, mape = trainer.baseline()
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")
    if args.test:
        mae, mape = trainer.inference()
        print(f"MAE: {mae}")
        print(f"MAPE: {mape}")
    else:
        print("Training one-d CNN is started")
        trainer.regression()
