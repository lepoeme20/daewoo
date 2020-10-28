import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# =============================================================================
# def get_dataloader(csv_path, iid=False):
#     df = pd.read_csv(csv_path)
# 
#     # transform = transforms.Compose([
#     #     transforms.Resize((32,32)),
#     #     transforms.ToTensor(),
#     # ])
# 
#     if iid:
#         # i.i.d condition
#         trn, dev, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
#     else:
#         # time series condition
#         trn, dev, tst = np.split(df, [int(.8*len(df)), int(.9*len(df))])
# 
#     trn_dataset = BuildDataset(trn)
#     dev_dataset = BuildDataset(dev)
#     tst_dataset = BuildDataset(tst)
#     
#     rand_sampler_tr = RandomSampler(trn_dataset,replacement=True,num_samples= np.int(len(trn_dataset)*0.01))
#     rand_sampler_val = RandomSampler(dev_dataset,replacement=True,num_samples= np.int(len(dev_dataset)*0.01))
#     
#     
#     trn_dataloader = torch.utils.data.DataLoader(
#         trn_dataset, batch_size=32, shuffle=False, num_workers=0,sampler= rand_sampler_tr
#     )
#     dev_dataloader = torch.utils.data.DataLoader(
#         dev_dataset, batch_size=32, shuffle=False, num_workers=0,sampler= rand_sampler_val
#     )
#     tst_dataloader = torch.utils.data.DataLoader(
#         tst_dataset, batch_size=32, shuffle=False, num_workers=0
#     )
# 
#     return trn_dataloader, dev_dataloader, tst_dataloader
# =============================================================================



class BuildDataset(Dataset):
    def __init__(self, df):
        self.img_path = df['image'].values
        self.labels = df['label'].values
        #self.transforms = transforms
        self.transforms  = transforms.Compose([
                    transforms.Resize((32,32)),
                    transforms.ToTensor(),
                        ]) 
        
        self.img_path1 = [] # label high ratio 
        self.img_path2 = [] # label low  ratio 
        self.label_path1 = []
        self.label_path2 = []
        self.threshold = 2.5 #일단 해두고 수정
        
        for i,label_value in enumerate((self.labels)):
            if label_value > self.threshold : # CHANGE FOR ENHANCING
                self.label_path1.append(self.labels[i])
                self.img_path1.append(self.img_path[i])
                
            else :
                self.label_path2.append(self.labels[i])
                self.img_path2.append(self.img_path[i])
        self.data_rate = (len(self.img_path1)/(len(self.img_path1)+len(self.img_path2)))
        
    def __len__(self):
        return len(self.img_path1) + len(self.img_path2)

    def __getitem__(self, idx):
        
        # 확률적으로 label 크기 에 따른 balanced 작업 수행
        if np.random.choice(2, 1, p=[1-self.data_rate, self.data_rate]) == 0:
            #idx = idx % len(self.img_path1)
            idx = np.random.randint(len(self.img_path1))
            img_path = self.img_path1[idx]
            label_path = self.label_path1[idx]
            
        else :
            
            idx = np.random.randint(len(self.img_path2))
            img_path = self.img_path2[idx]
            label_path = self.label_path2[idx]
        
        
        frame = Image.open(img_path)
        label = torch.tensor(label_path, dtype=torch.float)

        return self.transforms(frame), label
    
    