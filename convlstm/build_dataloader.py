import pandas as pd
import torch
from build_dataset import BuildDataset


def get_dataloader(use_weather_1, weather_1_csv_path, weather_1_root_img_path,
                   use_weather_4, weather_4_csv_path, weather_4_root_img_path,
                   use_total_phase, weather_total_phase_csv_path,
                   use_time_phase, img_split_type, iid, batch_size):

    # weather 1, 4 섞어서 분할한 csv file 사용시 
    if use_total_phase:
        df = pd.read_csv(weather_total_phase_csv_path)
        df["image"] = df["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_1/',
                                              weather_1_root_img_path)
        df["image"] = df["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_4/',
                                              weather_4_root_img_path)                                      
    else:
        # weather 1차 데이터 사용시 불러오기
        if use_weather_1:
            df1 = pd.read_csv(weather_1_csv_path)
            df1["image"] = df1["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_1/',
                                                    weather_1_root_img_path)
        else:
            df1 = pd.DataFrame()

        # weather 4차 데이터 사용시 불러오기
        if use_weather_4:
            df4 = pd.read_csv(weather_4_csv_path)
            df4["image"] = df4["image"].str.replace('/media/lepoeme20/Data/projects/daewoo/use_data/weather_4/',
                                                    weather_4_root_img_path)
        else:
            df4 = pd.DataFrame()

        df = pd.concat([df1, df4], ignore_index=True)

    if use_total_phase:
        trn = df.loc[df["total_phase"] == "train"]
        dev = df.loc[df["total_phase"] == "dev"]
        tst = df.loc[df["total_phase"] == "test"]
    else:
        if use_time_phase:
            trn = df.loc[df["time_phase"] == "train"]
            dev = df.loc[df["time_phase"] == "dev"]
            tst = df.loc[df["time_phase"] == "test"]
        else:
            # i.i.d condition
            trn = df.loc[df[f"iid_phase_{iid}"] == "train"]
            dev = df.loc[df[f"iid_phase_{iid}"] == "dev"]
            tst = df.loc[df[f"iid_phase_{iid}"] == "test"]

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
