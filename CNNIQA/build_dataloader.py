import torch
from build_dataset import BuildDataset
import pandas as pd


def get_dataloader(csv_path, root_img_path, batch_size, img_size, P=3, Q=3, C=1):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace('/hdd/data/daewoo/kisang1/', root_img_path)

    # i.i.d condition
    trn = df.loc[df["iid_phase"] == "train"]
    dev = df.loc[df["iid_phase"] == "dev"]
    tst = df.loc[df["iid_phase"] == "test"]

    trn_dataset = BuildDataset(trn, img_size, P, Q, C)
    dev_dataset = BuildDataset(dev, img_size, P, Q, C)
    tst_dataset = BuildDataset(tst, img_size, P, Q, C)

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