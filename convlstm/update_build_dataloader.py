import pandas as pd
import torch
from build_dataset import BuildDataset


def get_dataloader(
    brave_csv_path, brave_root_img_path, use_time_phase, img_split_type, batch_size, group_num
    ):

    df = pd.read_csv(brave_csv_path)
    df["image"] = df["image"].str.replace(
        '/mnt/C83AFA0C3AF9F6F2/hyundai_brave/crop_data', brave_root_img_path)


    if use_time_phase:
        df = df[df['time_phase'] == 'test']
    else:
    #     # i.i.d condition
        df = df[df['iid_phase'] == 'test']

    df = df[df['group'] == group_num] 
    dataset = BuildDataset(df, img_split_type)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # dataloader output: (batch_size, 3, 1, 448, 448)
    return dataloader