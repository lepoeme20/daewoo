import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
import itertools 


class BuildDataset(Dataset):
    def __init__(self, df, transform, img_size, sampling_ratio, label_type):
        
        self.img_path = df['image'].values
        if label_type == 'height':
            self.labels = df['height'].values
        elif label_type == 'direction':
            self.labels = df['direction'].values
        elif label_type == 'period':
            self.labels = df['period'].values
        elif label_type == 'cls':
            self.height = df['height'].values
            self.labels = df['class_label'].values
            
        self.sampling_ratio = sampling_ratio
        r_ratio = 1/sampling_ratio
        use_idx = [ i for i in range(len(self.labels)) if i%r_ratio ==0]
        
        img_path_list = df.iloc[use_idx,df.columns=='image'].values.tolist()
        img_path_list = list(itertools.chain(*img_path_list))
        
        label_path_list = df.iloc[use_idx,df.columns=='height'].values.tolist()
        label_path_list = list(itertools.chain(*label_path_list))
        
        self.img_path = img_path_list
        self.labels = label_path_list
        
        self.transform = transform
        self.img_size = img_size
        self.label_type = label_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        frame = cv2.imread(self.img_path[idx])
        if len(frame.shape) == 3:
            frame = frame[:, :, 0]
        
        frame = self.image_split_concat(frame)
        
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
        
    def image_split_concat(self,frame):
        frame1 = frame[:,:448]
        frame2 = frame[:,448:448*2]
        frame3 = frame[:,448*2:448*3]
        new_frame = np.stack((frame1,frame2,frame3),axis=2)
        return new_frame
        
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
