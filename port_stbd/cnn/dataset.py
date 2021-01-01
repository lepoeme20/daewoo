import glob
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class WaveDataset(Dataset):
    def __init__(
        self,
        data: list,
        img_size: int = 32,
    ):
        super(WaveDataset, self).__init__()
        self.data = data
        self.img_size = img_size

    def get_transform(self, frame):
        """ per-image normalization """
        # z score scaling
        frame = frame.astype(np.float32)
        mean = frame.mean()
        std = frame.std()

        # normalize 진행
        frame -= mean
        frame /= std

        # Resize, 확대시 cv2.INTER_CUBIC
        frame = cv2.resize(
            frame, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        return transforms.ToTensor()(frame)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        img_path, label = pair["img"], pair["label"]

        frame = cv2.imread(img_path)
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]

        label = torch.tensor(label, dtype=torch.long)
        return self.get_transform(frame), label


def get_dataloaders(root: str, train_batch_size: int, eval_batch_size: int):
    classes = {"PORT": 0, "STBD": 1}

    pth_port = os.path.join(root, "PORT/day_time")
    pth_stbd = os.path.join(root, "STBD/day_time")

    # aggregate all data
    img_port, img_stbd = [], []
    for folder in os.listdir(pth_port):
        for img in glob.glob(os.path.join(pth_port, folder, "*.jpg")):
            img_port.append({"img": img, "label": classes["PORT"]})  # label 0
    for folder in os.listdir(pth_stbd):
        for img in glob.glob(os.path.join(pth_stbd, folder, "*.jpg")):
            img_stbd.append({"img": img, "label": classes["STBD"]})  # label 1

    img_stbd = random.sample(img_stbd, 10000)  # sampling 10000 at STBD for balance
    data = img_port + img_stbd
    random.shuffle(data)

    train_data_size = int(len(data) * 0.7)
    val_data_size = (len(data) - train_data_size) // 2

    train_data = data[:train_data_size]
    val_data = data[train_data_size : train_data_size + val_data_size]
    test_data = data[train_data_size + val_data_size :]

    train_dset = WaveDataset(train_data)
    val_dset = WaveDataset(val_data)
    test_dset = WaveDataset(test_data)

    return (
        DataLoader(train_dset, batch_size=train_batch_size, shuffle=True),
        DataLoader(val_dset, batch_size=eval_batch_size, shuffle=False),
        DataLoader(test_dset, batch_size=eval_batch_size, shuffle=False),
    )
