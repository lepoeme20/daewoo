import pandas as pd
import torch
from build_dataset import BuildDataset


def get_dataloader(csv_path, root_img_path, batch_size):
    df = pd.read_csv(csv_path)
    df["image"] = df["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_4/', root_img_path)

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