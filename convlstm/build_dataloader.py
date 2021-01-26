import pandas as pd
import torch
from build_dataset import BuildDataset


def get_dataloader(csv_path, root_img_path, batch_size, add_csv_path=None, add_root_img_path=None):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_4/', root_img_path)
    
    if add_csv_path is not None and add_root_img_path is not None:
        add_df = pd.read_csv(add_csv_path)
        add_df["image"] = add_df["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_1/', add_root_img_path)
        df = pd.concat([df, add_df], ignore_index=True)

    # i.i.d condition
    trn = df.loc[df["iid_phase"] == "train"]
    dev = df.loc[df["iid_phase"] == "dev"]
    tst = df.loc[df["iid_phase"] == "test"]

    trn_dataset = BuildDataset(trn)
    dev_dataset = BuildDataset(dev)
    tst_dataset = BuildDataset(tst)

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