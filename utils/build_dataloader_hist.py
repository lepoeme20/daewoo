# Main module
import torch
import torchvision
import torchvision.transforms as transforms
from utils.build_dataset_hist import BuildDataset

import pandas as pd
import numpy as np
from torch.utils.data.sampler import SequentialSampler, RandomSampler , SubsetRandomSampler

def get_dataloader(csv_path, batch_size, iid=False, transform=2, img_size=224,hist_option='he'):
    #csv_path= './weather_data_label.csv'
    df = pd.read_csv(csv_path)

    if iid:
        # i.i.d condition
        #trn, dev, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        trn = df.loc[df['iid_phase']=='train',:]
        dev = df.loc[df['iid_phase']=='dev',:]
        tst = df.loc[df['iid_phase']=='test',:]
    else:
        # time series condition
        #trn, dev, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])
        trn = df.loc[df['time_phase']=='train',:]
        dev = df.loc[df['time_phase']=='dev',:]
        tst = df.loc[df['time_phase']=='test',:]
         
    trn_dataset = BuildDataset(trn, transform, img_size, 0.1,hist_option)
    dev_dataset = BuildDataset(dev, transform, img_size, 0.1,hist_option)
    tst_dataset = BuildDataset(tst, transform, img_size, 0.1,hist_option)
   
    #rand_sampler_tr = RandomSampler(trn_dataset,replacement=True,num_samples= np.int(len(trn_dataset)*0.1))
    rand_sampler_tr = RandomSampler(trn_dataset,replacement=True,num_samples= np.int(len(trn_dataset)))
    rand_sampler_val = RandomSampler(dev_dataset,replacement=True,num_samples= np.int(len(dev_dataset)))
    
    
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=8 , 
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8, 
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
    )

    return trn_dataloader, dev_dataloader, tst_dataloader

