import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from cnn_classification.get_class import class_label

class BuildDataset(Dataset):
    def __init__(self, df, transform, img_size, label_type, cls_range=1):
        self.img_path = df['image'].values
        if label_type == 'height':
            self.labels = df['height'].values
        elif label_type == 'direction':
            self.labels = df['direction'].values
        elif label_type == 'period':
            self.labels = df['period'].values
        elif label_type == 'cls':
            if cls_range != 0:
                df = class_label().generate_class(df, 'label', cls_range*10)
            self.height = df['height'].values
            self.labels = (
                df["label"].values
                if cls_range == 0
                else df[f"class_label_{cls_range*10}"].values
            )

        self.transform = transform
        self.img_size = img_size
        self.label_type = label_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = cv2.imread(self.img_path[idx])
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]
        if self.label_type == 'cls':
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            height = torch.tensor(self.height[idx], dtype=torch.float)
            return self.get_transform(frame), label, height
        else:
            if self.label_type == 'direction':
                label = torch.tensor(self.labels[idx] // 3, dtype=torch.long)
            else:
                label = torch.tensor(self.labels[idx], dtype=torch.float)
            return self.get_transform(frame), label

    def get_transform(self, frame):
        if self.transform == 0:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
            return transform(frame)
        elif self.transform == 1:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3352], std=[0.0647]),
            ])
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
                dsize=(self.img_size, self.img_size),
                interpolation=cv2.INTER_AREA
                )

            frame = transforms.ToPILImage()(frame)
            frame = transforms.RandomHorizontalFlip()(frame)

            return transforms.ToTensor()(np.array(frame))
