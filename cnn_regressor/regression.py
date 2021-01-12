import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

# from resnet import resnet34

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.functions as F
from utils.build_dataloader import get_dataloader

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
        self.pretrain = args.pretrain
        self.model = MODELS[args.model_name](pretrained=args.imagenet)
        self.model = nn.Sequential(
            self.model,
            nn.Linear(1000, 1),
        )
        self.dataset = args.dataset

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=20
            )

        # data loader
        self.trn_loader, self.dev_loader, self.tst_loader = get_dataloader(
                csv_path=args.csv_path,
                batch_size=args.batch_size,
                label_type=args.label_type,
                iid=args.iid,
                transform=args.norm_type,
                img_size=args.img_size
        )

        # set path
        self.model_path = f'./cnn_regressor/best_model/{args.dataset}/{args.label_type}/norm_{args.norm_type}/{args.data_type}'
        os.makedirs(self.model_path, exist_ok=True)

    def pretraining(self):
        # model
        self.model.to(self.device)
        # initial dev loss
        best_loss = 1000.
        # model path
        pretrained_path = os.path.join(self.model_path, 'pretrained_model.pt')
        # criterion
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            print("")
            self._train_loop(epoch, criterion)
            print("")
            best_loss = self._dev_loop(epoch, criterion, best_loss, pretrained_path)

    def regression(self):
        # model
        self.model.to(self.device)
        # initial dev loss
        best_loss = 1000.
        # model path
        pretrained_path = os.path.join(self.model_path, 'pretrained_model.pt')
        # regressor_path = os.path.join(self.model_path, 'regressor.pt')
        regressor_path = os.path.join(self.model_path, 'regressor_residual.pt')

        # criterion
        criterion = nn.MSELoss()

        # load pretrained model
        # checkpoint = torch.load(pretrained_path)
        # if isinstance(self.model, nn.DataParallel):
        #     self.model.module.load_state_dict(checkpoint['model_state_dict'])
        # else:
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.fc = nn.Linear(self.model.fc.in_features, 1).to(self.device)

        for epoch in range(self.epochs):
            print("")
            self._train_loop(epoch, criterion)
            print("")
            best_loss = self._dev_loop(epoch, criterion, best_loss, regressor_path)

    def _train_loop(self, epoch, criterion):
        total_step = 0
        for step, (inputs, labels) in enumerate(self.trn_loader):
            self.model.train()
            total_step += 1

            inputs = inputs.repeat(1,3,1,1) ##
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.pretrain:
                labels = F.get_cls_label(labels, self.dataset).to(self.device)

            # output, _ = self.model(inputs)
            output = self.model(inputs)
            loss = criterion(torch.squeeze(output), labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #################### Logging ###################
            print("[Train] Epoch {}/{} Batch {}/{} Loss: {:.4f}".format(
                epoch+1, args.epochs, step+1, len(self.trn_loader), loss.item()
                ), end='\r')

    def _dev_loop(self, epoch, criterion, best_loss, save_path):
        _dev_loss, dev_loss = 0., 0.
        for idx, (inputs, labels) in enumerate(self.dev_loader, 0):
            self.model.eval()
            inputs = inputs.repeat(1,3,1,1) ##
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.pretrain:
                labels = F.get_cls_label(labels, self.dataset).to(self.device)

            with torch.no_grad():
                # Cross Entropy loss
                # logit, _ = self.model(inputs)
                logit = self.model(inputs)
                loss = criterion(torch.squeeze(logit), labels)

                # Loss
                _dev_loss += loss
                dev_loss = _dev_loss/(idx+1)

                print('[Dev] {}/{} Loss: {:.3f}'.format(
                    idx+1, len(self.dev_loader), dev_loss), end='\r')

        self.scheduler.step(dev_loss)

        if dev_loss < best_loss:
            print("Loss: {:.4f}".format(dev_loss))
            best_loss = dev_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'trained_epoch': epoch,
            }, save_path)
        return best_loss


    def inference(self):
        self.model.to(self.device)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1).to(self.device)
        loss_mae = 0.0
        loss_mape = 0.0

        # load the saved regressor
        regressor_path = os.path.join(self.model_path, 'regressor.pt')
        checkpoint = torch.load(regressor_path)
        # check DataParallel
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.tst_loader),
                desc="test steps",
                total=len(self.tst_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)
                img = img.repeat(1,3,1,1)

                # output, _ = self.model(img)
                output = self.model(img)
                maeloss = self.MAE(output.squeeze(), label)
                mapeloss = self.MAPE(output.squeeze(), label)

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
                enumerate(self.tst_loader),
                desc="test step",
                total=len(self.tst_loader)
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


if __name__ == '__main__':
    # Set random seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='weather',
        help="csv file path"
    )
    parser.add_argument(
        "--norm-type", type=int, choices=[0, 1, 2],
        help="0: ToTensor, 1: Ordinary image normalizaeion, 2: Image by Image normalization"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Batch size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="Batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=22, help="seed number"
    )
    parser.add_argument(
        "--label-type", type=int, choices=[0, 1, 2],
        help="0: Height, 1: Direction, 2: Period"
    )
    parser.add_argument(
        "--iid", action="store_true", default=False, help="use argument for iid condition"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Perform Test only"
    )
    parser.add_argument(
        "--pretrain", action="store_true", default=False, help="Train pretrained model"
    )
    parser.add_argument(
        "--baseline", action="store_true", default=False, help="Train pretrained model"
    )
    parser.add_argument(
        "--imagenet", action="store_true", default=False, help="ImageNet pretrained"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name (ResNet18, DenseNet, GoogLenet, VGG)",
        choices=["ResNet18", "DenseNet", "GoogLenet", "VGG"],
    )
    parser.add_argument("--num-classes", type=int, default=1)
    args = parser.parse_args()

    args.csv_path = f'./preprocessing/{args.dataset}_data_label.csv'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.data_type = 'iid' if args.iid else 'time'
    if args.label_type == 0:
        print(" Set label as height")
        args.label_type = 'height'
    elif args.label_type == 1:
        print(" Set label as direction")
        args.label_type = 'direction'
    elif args.label_type == 2:
        print(" Set label as period")
        args.label_type = 'period'

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
        if args.pretrain:
            print("Pretrain with classification task is started")
            trainer.pretraining()
        else:
            print("Training CNN Regressor is started")
        trainer.regression()
