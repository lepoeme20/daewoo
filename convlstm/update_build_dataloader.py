import pandas as pd
import torch
import os
from build_dataset import BuildDataset
import IPython


def get_dataloader(
    brave_csv_path, brave_root_img_path, use_time_phase, img_split_type, batch_size, use_total_phase, use_prediction, group_num, save_path
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

    if use_prediction:
        print('**Using inference result as true label**')
        if use_total_phase:
            file_name = 'result_source_target_total_phase_beforeupdate{}th.csv'.format(group_num)
        else:
            if use_time_phase:
                file_name = 'result_source_{}_target_{}_beforeupdate{}th.csv'.format('time_phase', 'brave', group_num)


        df_pred = pd.read_csv(os.path.join(save_path,file_name))
        df['height'] = df_pred['pred'].values

    dataset = BuildDataset(df, img_split_type)


    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # dataloader output: (batch_size, 3, 1, 448, 448)
    return dataloader