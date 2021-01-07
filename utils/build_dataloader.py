import torch
from utils.build_dataset import BuildDataset
import pandas as pd


def get_dataloader(
    csv_path, batch_size, label_type, iid=False, transform=2, img_size=224, cls_range=1
):
    df = pd.read_csv(csv_path)

    if iid:
        # i.i.d condition
        trn = df.loc[df["iid_phase"] == "train"]
        dev = df.loc[df["iid_phase"] == "dev"]
        tst = df.loc[df["iid_phase"] == "test"]
    else:
        # time series condition
        trn = df.loc[df["time_phase"] == "train"]
        dev = df.loc[df["time_phase"] == "dev"]
        tst = df.loc[df["time_phase"] == "test"]

    trn_dataset = BuildDataset(trn, transform, img_size, label_type, cls_range)
    dev_dataset = BuildDataset(dev, transform, img_size, label_type, cls_range)
    tst_dataset = BuildDataset(tst, transform, img_size, label_type, cls_range)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return trn_dataloader, dev_dataloader, tst_dataloader