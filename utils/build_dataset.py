import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage
import cv2
import numpy as np

class BuildDataset(Dataset):
    def __init__(self, df, transform, img_size):
        self.img_path = df['image'].values
        self.labels = df['label'].values
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return self.get_transform(frame), label

    def get_transform(self, frame):
        if self.transform == 0:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
        elif self.transform == 1:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3352], std=[0.0647]),
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
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
        return transform(frame)
