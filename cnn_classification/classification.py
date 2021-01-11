import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader
import utils.functions as pred2height
from cnn_classification.model_utils import softXEnt

MODELS = {
    "ResNet18": torchvision.models.resnet18,
    "DenseNet": torchvision.models.densenet121,
    "GoogLenet": torchvision.models.googlenet,
    "VGG": torchvision.models.vgg16,
}


class Trainer:
    def __init__(self, args):
        self.device = args.device
        self.epochs = args.epochs
        if args.model_name in ["ResNet18", "GoogLenet"]:
            self.model = MODELS[args.model_name](
                pretrained=False, num_classes=args.num_classes
            )
            # self.model.fc = nn.Linear(self.model.fc.in_features, args.num_classes).to(
            #     self.device
            # ) ## TODO: add Global Average Pooling option
        elif args.model_name == "DenseNet":
            self.model = MODELS[args.model_name](
                pretrained=False, num_classes=args.num_classes
            )
            # self.model.classifier = nn.Linear(
            #     self.model.classifier.num_features, args.num_classes
            # ).to(self.device)
        elif args.model_name == "VGG":
            self.model = MODELS[args.model_name](
                pretrained=False, num_classes=args.num_classes
            )

        self.dataset = args.dataset
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10
        )
        self.loss = args.loss
        self.window_size = args.window
        self.num_classes = args.num_classes
        self.label_range = args.label_range
        # data loader
        self.trn_loader, self.dev_loader, self.tst_loader = get_dataloader(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            label_type=args.label_type,
            iid=args.iid,
            transform=args.norm_type,
            img_size=args.img_size,
            cls_range=args.label_range,
        )

        # set path
        self.model_path = (
            f"./cnn_classification/best_model/{args.dataset}/{args.label_type}/norm_{args.norm_type}/{args.data_type}"
            if args.label_range == 0
            else f"./cnn_classification/best_model/{args.dataset}/{args.label_type}/norm_{args.norm_type}/{args.data_type}/{args.label_range}0"
        )
        os.makedirs(self.model_path, exist_ok=True)

    def classification(self):
        # model
        self.model.to(self.device)
        # initial dev loss
        best_loss = 1000.0
        # model path
        model_path = os.path.join(self.model_path, f"{args.model_name}.pt")
        # criterion
        if self.loss == "ce":
            criterion = nn.CrossEntropyLoss()
        elif self.loss == "soft_ce":
            criterion = softXEnt

        outer = tqdm(total=args.epochs, desc="Epoch", position=0, leave=False)
        for epoch in range(self.epochs):
            self._train_loop(epoch, criterion)
            best_loss = self._dev_loop(epoch, criterion, best_loss, model_path)
        outer.update(1)

    def _train_loop(self, epoch, criterion):
        total_step = 0
        trn_loss_log = tqdm(total=0, position=2, bar_format="{desc}")
        train = tqdm(total=len(self.trn_loader), desc="Steps", position=1, leave=False)
        for step, (inputs, labels, _) in enumerate(self.trn_loader):
            self.model.train()
            total_step += 1

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.repeat(1, 3, 1, 1)

            output = self.model(inputs)

            if self.loss == "ce":
                loss = (
                    criterion(torch.squeeze(output), labels)
                    if args.model_name != "GoogLenet"
                    else criterion(torch.squeeze(output[0]), labels)
                )
            elif self.loss == "soft_ce":
                loss = (
                    criterion(
                        torch.squeeze(output),
                        labels,
                        class_num=self.num_classes,
                        window_size=self.window_size,
                    )
                    if args.model_name != "GoogLenet"
                    else criterion(
                        torch.squeeze(output[0]),
                        labels,
                        class_num=self.num_classes,
                        window_size=self.window_size,
                    )
                )

            """    
            loss = (
                criterion(torch.squeeze(output), labels)
                if args.model_name != "GoogLenet"
                else criterion(torch.squeeze(output[0]), labels)
            )  # googlenet has 3 outputs with aux-target in training step
            """

            self.optimizer.zero_grad()
            loss.backward()
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
        for idx, (inputs, labels, _) in enumerate(self.dev_loader, 0):
            self.model.eval()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.repeat(1, 3, 1, 1)

            with torch.no_grad():
                # Cross Entropy loss
                logit = self.model(inputs)

                if self.loss == "ce":
                    loss = criterion(torch.squeeze(logit), labels)
                elif self.loss == "soft_ce":
                    loss = criterion(
                        torch.squeeze(logit),
                        labels,
                        class_num=self.num_classes,
                        window_size=self.window_size,
                    )

                # Loss
                _dev_loss += loss
                dev_loss = _dev_loss / (idx + 1)

                dev_loss_log.set_description_str(
                    f"[Dev] {idx+1}/{len(self.dev_loader)} Loss: {dev_loss:.4f}"
                )
                dev.update(1)

        self.scheduler.step(dev_loss)

        if dev_loss < best_loss:
            best_epoch_log.set_description_str(f"Loss: {dev_loss:.4f}")
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
        self.model.to(self.device)
        loss_mae = 0.0
        loss_mape = 0.0

        # load the saved model
        model_path = os.path.join(self.model_path, f"{args.model_name}.pt")
        checkpoint = torch.load(model_path)
        # check DataParallel
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.tst_loader),
                desc="test steps",
                total=len(self.tst_loader),
            ):
                img, _, height = map(lambda x: x.to(self.device), batch)
                img = img.repeat(1, 3, 1, 1)

                output = self.model(img)
                _, pred = torch.max(output, 1)
                output = pred2height(pred, self.label_range)

                maeloss = self.MAE(output.squeeze(), height)
                mapeloss = self.MAPE(output.squeeze(), height)

                loss_mae += maeloss.item()
                loss_mape += mapeloss.item()

        loss_mae /= step + 1
        loss_mape /= step + 1

        return loss_mae, loss_mape

    def baseline(self):
        with torch.no_grad():
            trn_mean = torch.zeros((len(self.trn_loader), 1), device=self.device)
            for idx, (_, _, height) in tqdm(
                enumerate(self.trn_loader),
                desc="compute train mean",
                total=len(self.trn_loader),
            ):
                batch_height = height.to(self.device)
                trn_mean[idx] = torch.mean(batch_height).item()

            trn_mean = torch.mean(trn_mean)
            mae = 0.0
            mape = 0.0
            for step, (_, _, height) in tqdm(
                enumerate(self.tst_loader), desc="test step", total=len(self.tst_loader)
            ):
                y_tst = height.to(self.device)
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
    parser.add_argument("--dataset", type=str, default="weather", help="csv file path")
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
    parser.add_argument("--lr", type=float, default=0.05, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Batch size")
    parser.add_argument("--seed", type=int, default=22, help="seed number")
    parser.add_argument(
        "--label-type",
        type=int,
        default=3,
        help="0: Height, 1: Direction, 2: Period, 3: Classification label",
    )
    parser.add_argument(
        "--iid",
        action="store_true",
        default=False,
        help="use argument for iid condition",
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Perform Test only"
    )
    parser.add_argument(
        "--baseline", action="store_true", default=False, help="Train pretrained model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name (ResNet18, DenseNet, GoogLenet, VGG)",
        choices=["ResNet18", "DenseNet", "GoogLenet", "VGG"],
    )
    parser.add_argument(  ### TODO
        "--classifier",
        type=str,
        help="classifier after feature extraction (linear, gap)",
        choices=["linear", "gap"],
    )
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "soft_ce"])
    parser.add_argument("--window", type=int, default=2, help="window size in soft ce")
    parser.add_argument("--num-classes", type=int, help="number of classes", default=10)
    parser.add_argument(
        "--label-range",
        type=int,
        choices=[0, 1, 2],
        help="0: minmax / 1: 10cm range / 2: 20cm range",
    )

    args = parser.parse_args()

    args.csv_path = os.path.join(args.root_csv_path, f"{args.dataset}_data_label.csv")
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.data_type = "iid" if args.iid else "time"

    if args.label_type == 0:
        print(" Set label as height")
        args.label_type = "height"
    elif args.label_type == 1:
        print(" Set label as direction")
        args.label_type = "direction"
    elif args.label_type == 2:
        print(" Set label as period")
        args.label_type = "period"
    elif args.label_type == 3:
        print(" Set label as classification")
        args.label_type = "cls"

    if args.label_range == 0:
        print("Use min-median-max range")
    elif args.label_range == 1:
        args.num_classes = 20
        print("Use 10cm range")
    elif args.label_range == 2:
        print("Use 20cm range")

    print("Modifying img path in csv file...")
    label_df = pd.read_csv(args.csv_path)
    col_img_pth = label_df["image"]
    folder_idx = col_img_pth[0].find("data_crop")  # 'data_crop' is shared folder name
    pth_before = col_img_pth[0][:folder_idx]
    assert args.root_img_path[-1] == "/", "이미지 경로 마지막을 / 으로 끝나게 해 주세요~!"
    label_df["image"] = label_df["image"].str.replace(pth_before, args.root_img_path)
    label_df.to_csv(args.csv_path, index=False)
    print("Done!")

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
        print("Training CNN is started")
        trainer.classification()