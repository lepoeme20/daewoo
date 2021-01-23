
import os
import sys
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class BuildDataset(Dataset):
    """
    Dataset for ConvLSTM: 3 frames from rectangular image construct 1 clip
    """
    def __init__(self, df):
        self.img_path = df["image"].values
        self.labels = df["height"].values

        self.img_width = 1344
        self.img_height = 448

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])  # (448, 1344, 3)
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        if len(frame.shape) == 3:
            frame = frame[:, :, 0]  # (448, 1344)

        frame = self.get_transform(frame)

        # convert rectangular into clip
        frame_1 = frame[:, :, : self.img_height]
        frame_2 = frame[:, :, self.img_height : 2 * self.img_height]
        frame_3 = frame[:, :, 2 * self.img_height :]
        clip = torch.stack((frame_1, frame_2, frame_3), axis=0)  # (3, 1, 448, 448)
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