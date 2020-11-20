import torch
from utils.build_dataset import BuildDataset
import pandas as pd
import numpy as np

def get_dataloader(csv_path, batch_size, dtype, iid=False, transform=2, img_size=224):
    df = pd.read_csv(csv_path)

    if iid:
        # i.i.d condition
        unique_id = np.unique(df['label_idx'])
        np.random.shuffle(unique_id)
        trn_idx, dev_idx, tst_idx = np.split(
            unique_id, [int(.6*len(unique_id)), int(.8*len(unique_id))]
            )
        trn = df.loc[df['label_idx'].isin(trn_idx)]
        dev = df.loc[df['label_idx'].isin(dev_idx)]
        tst = df.loc[df['label_idx'].isin(tst_idx)]
    else:
        # time series condition
        trn, dev, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])

    trn_dataset = BuildDataset(trn, transform, img_size, dtype)
    dev_dataset = BuildDataset(dev, transform, img_size, dtype)
    tst_dataset = BuildDataset(tst, transform, img_size, dtype)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=64
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=64
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=64
    )

    return trn_dataloader, dev_dataloader, tst_dataloader