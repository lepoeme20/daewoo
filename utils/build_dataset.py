import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np

class BuildDataset(Dataset):
    def __init__(self, df, transform):
        self.img_path = df['image'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return self.get_transform(frame), label

    def get_transform(self, frame):
        if self.transform == 0:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.transform == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
        else:
            # z score scaling
            frame = frame.astype(np.float32)
            mean = frame.mean()
            std = frame.std()

            # normalize 진행
            frame -= mean
            frame /= std
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        return transform(frame)