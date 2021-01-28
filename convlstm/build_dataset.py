import os
import sys
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class BuildDataset(Dataset):
    """
    Dataset for ConvLSTM: 3 frames from rectangular image construct 1 clip
    """
    def __init__(self, df, img_split_type):
        self.img_path = df["image"].values
        self.labels = df["height"].values
        self.img_split_type = img_split_type

        self.img_width = 1344
        self.img_height = 448

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])  # (448, 1344, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        if len(frame.shape) == 3:
            frame = frame[:, :, 0]  # (448, 1344)

        frame = self.get_transform(frame)  # (1, 448, 1344)

        if self.img_split_type == 0:
            step = self.img_width // 3
            size = self.img_width // 3
            limit = self.img_width
        elif self.img_split_type == 1:
            step = self.img_width // 6
            size = self.img_width // 3
            limit = self.img_width
        else:
            step = self.img_height // 4
            size = self.img_height // 4
            limit = self.img_height

        i = 0
        clip = torch.Tensor()
        while i * step + size <= limit:
            if self.img_split_type == 2:
                _frame = frame[:, i * step: i * step + size, :]
            else:
                _frame = frame[:, :, i * step: i * step + size]
            # img_split_type: 0 -> (3, 1, 448, 448) | 1 -> (5, 1, 448, 448) | 2 -> (4, 1, 112, 1344)
            clip = torch.cat([clip, _frame.unsqueeze(0)], axis=0)
            i += 1

        return clip, label

    def get_transform(self, frame):
        # z score scaling
        frame = frame.astype(np.float32)
        mean = frame.mean()
        std = frame.std()
        if std == 0:
            std = 1

        # normalize
        frame -= mean
        frame /= std

        frame = transforms.ToPILImage()(frame)
        return transforms.ToTensor()(np.array(frame))