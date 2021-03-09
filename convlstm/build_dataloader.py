import pandas as pd
import torch
from build_dataset import BuildDataset


def get_dataloader(
    brave_csv_path, brave_root_img_path, use_time_phase, img_split_type, batch_size
    ):

    df = pd.read_csv(brave_csv_path)
    df["image"] = df["image"].str.replace(
        '/media/lepoeme20/Data/projects/daewoo/hyundai_brave', brave_root_img_path)

    if use_time_phase:
        trn = df.loc[df["time_phase"] == "train"]
        dev = df.loc[df["time_phase"] == "dev"]
        tst = df.loc[df["time_phase"] == "test"]
    else:
        # i.i.d condition
        trn = df.loc[df[f"iid_phase"] == "train"]
        dev = df.loc[df[f"iid_phase"] == "dev"]
        tst = df.loc[df[f"iid_phase"] == "test"]

    trn_dataset = BuildDataset(trn, img_split_type)
    dev_dataset = BuildDataset(dev, img_split_type)
    tst_dataset = BuildDataset(tst, img_split_type)

    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    tst_dataloader = torch.utils.data.DataLoader(
        tst_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    # dataloader output: (batch_size, 3, 1, 448, 448)
    return trn_dataloader, dev_dataloader, tst_dataloader
