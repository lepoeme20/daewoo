import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from cnn_classification.get_class import class_label
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import RandomHorizontalFlip


class BuildDataset(Dataset):
    """Dataset for ConvLSTM

    Rectangular image is divided into 3 clips(channels)
    """

    def __init__(self, df, transform, label_type, cls_range=1):
        self.img_path = df["image"].values
        if label_type == "height":
            self.labels = df["height"].values
        elif label_type == "direction":
            self.labels = df["direction"].values
        elif label_type == "period":
            self.labels = df["period"].values
        elif label_type == "cls":
            if cls_range != 0:
                n_cls = 10 if cls_range == 2 else 20  # 10cm range means 20 classes
                df = class_label().generate_class(df, "label", n_cls)
            self.height = df["height"].values
            self.labels = (
                df["label"].values
                if cls_range == 0
                else df[f"class_label_{n_cls}"].values
            )

        self.transform = transform
        self.img_width = 1344
        self.img_height = 448
        self.label_type = label_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])  # (448, 1344, 3)

        if len(frame.shape) == 3:
            frame = frame[:, :, 0]  # (448, 1344)

        # convert rectangular into clip
        frame_1 = frame[:, : self.img_height]
        frame_2 = frame[:, self.img_height : 2 * self.img_height]
        frame_3 = frame[:, 2 * self.img_height :]
        frame = np.stack((frame_1, frame_2, frame_3), axis=2)  # (448, 448, 3)

        if self.label_type == "cls":
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            height = torch.tensor(self.height[idx], dtype=torch.float)
            return self.get_transform(frame), label, height
        else:
            if self.label_type == "direction":
                label = torch.tensor(self.labels[idx] // 3, dtype=torch.long)
            else:
                label = torch.tensor(self.labels[idx], dtype=torch.float)

            return self.get_transform(frame), label

    def get_transform(self, frame):
        if self.transform == 0:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Resize((self.img_height, self.img_height)),
                    transforms.ToTensor(),
                ]
            )
            return transform(frame)
        elif self.transform == 1:
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Resize((self.img_height, self.img_height)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.3352], std=[0.0647]),
                ]
            )
            return transform(frame)
        else:
            # z score scaling
            frame = frame.astype(np.float32)
            mean = frame.mean()
            std = frame.std()

            # normalize 진행
            frame -= mean
            frame /= std

            # Resize, 확대시 cv2.INTER_CUBIC
            frame = cv2.resize(
                frame,
                dsize=(self.img_height, self.img_height),
                interpolation=cv2.INTER_AREA,
            )

            frame = frame.astype(np.uint8)  ## 21.01.22 added
            frame = transforms.ToPILImage()(frame)
            frame = transforms.RandomHorizontalFlip()(frame)

            return transforms.ToTensor()(np.array(frame))


def get_dataloader(
    csv_path, batch_size, label_type, iid=False, transform=2, cls_range=1
):
    df = pd.read_csv(csv_path)

    if iid:
        # i.i.d condition
        trn = df.loc[df["iid_phase"] == "train"]
        dev = df.loc[df["iid_phase"] == "dev"]
        tst = df.loc[df["iid_phase"] == "test"]
    else:
        # time series condition
        trn = df.loc[df["time_phase"] == "train"]
        dev = df.loc[df["time_phase"] == "dev"]
        tst = df.loc[df["time_phase"] == "test"]

    trn_dataset = BuildDataset(trn, transform, label_type, cls_range)
    dev_dataset = BuildDataset(dev, transform, label_type, cls_range)
    tst_dataset = BuildDataset(tst, transform, label_type, cls_range)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # dataloader output: (batch_size, 3, 448, 448)
    return trn_dataloader, dev_dataloader, tst_dataloader
