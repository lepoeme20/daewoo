import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from scipy.signal import convolve2d


class BuildDataset(Dataset):
    def __init__(self, df, img_size, P, Q, C):
        self.img_path = df['image'].values
        self.labels = df['height'].values
        self.img_size = img_size
        self.P = P
        self.Q = Q
        self.C = C
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])[:, :, 0]
        frame = self.transforms(frame)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return self.LocalNormalization(frame[0]), label

    def LocalNormalization(self, frame):
        kernel = np.ones((self.P, self.Q)) / (self.P * self.Q)
        frame_mean = convolve2d(frame, kernel, boundary='symm', mode='same')
        frame_sm = convolve2d(np.square(frame), kernel, boundary='symm', mode='same')
        frame_std = np.sqrt(np.maximum(frame_sm - np.square(frame_mean), 0)) + self.C
        frame_ln = ((frame[0] - frame_mean) / frame_std).float().unsqueeze(0)
        return frame_ln
