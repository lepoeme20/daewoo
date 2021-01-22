import torch
from utils.build_dataset import BuildDataset
import pandas as pd

def get_dataloader(csv_path, batch_size, label_type, iid=False,
                   transform=2, img_size=224):
    df = pd.read_csv(csv_path)
    #df = label_df
    if iid:
        # i.i.d condition
        trn = df.loc[df['iid_phase'] == 'train']
        dev = df.loc[df['iid_phase'] == 'dev']
        tst = df.loc[df['iid_phase'] == 'test']
    else:
        # time series condition
        trn = df.loc[df['time_phase'] == 'train']
        dev = df.loc[df['time_phase'] == 'dev']
        tst = df.loc[df['time_phase'] == 'test']
        
    trn_dataset = BuildDataset(trn, transform, img_size,0.1, label_type)
    dev_dataset = BuildDataset(dev, transform, img_size,0.5, label_type)
    tst_dataset = BuildDataset(tst, transform, img_size,1, label_type)
        
   
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