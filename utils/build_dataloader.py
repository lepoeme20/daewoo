# Main module
import torch
import torchvision
import torchvision.transforms as transforms
from utils.build_dataset import BuildDataset
import pandas as pd
import numpy as np

def get_dataloader(csv_path, batch_size, iid=False, transform=2):
    df = pd.read_csv(csv_path)

    if iid:
        # i.i.d condition
        trn, dev, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    else:
        # time series condition
        trn, dev, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])

    trn_dataset = BuildDataset(trn, transform)
    dev_dataset = BuildDataset(dev, transform)
    tst_dataset = BuildDataset(tst, transform)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return trn_dataloader, dev_dataloader, tst_dataloader