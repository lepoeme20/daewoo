import glob
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

classes = {"PORT": 0, "STBD": 1}


class WaveDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: int = 32,
    ):
        super(WaveDataset, self).__init__()
        self.root = root
        self.pth_port = os.path.join(self.root, "PORT/day_time")
        self.pth_stbd = os.path.join(self.root, "STBD/day_time")
        self.img_size = img_size

        # aggregate all data
        img_port, img_stbd = [], []
        for folder in os.listdir(self.pth_port):
            for img in glob.glob(os.path.join(self.pth_port, folder, "*.jpg")):
                img_port.append({"img": img, "label": classes["PORT"]})  # label 0
        for folder in os.listdir(self.pth_stbd):
            for img in glob.glob(os.path.join(self.pth_stbd, folder, "*.jpg")):
                img_stbd.append({"img": img, "label": classes["STBD"]})  # label 1
        self.data = img_port + img_stbd
        random.shuffle(self.data)  # TODO: train/val/test 구분되어 있으면 지울 것

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


def get_dataloader(root: str, batch_size: int, shuffle: bool):
    dset = WaveDataset(root)
    return DataLoader(dset, batch_size=batch_size, shuffle=True)
