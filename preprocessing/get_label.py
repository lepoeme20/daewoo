import os
import pandas as pd
import numpy as np

def cal_time(time):
    full_time = ''.join(time[:10].split('-')) + ''.join(time[11:].split(':'))
    h = full_time[8:10]
    m = full_time[10:12]
    start_m = str(int(m)-2)
    end_m = str(int(m)+2)
    start_h, end_h = h, h
    if start_m < "0":
        start_m = str(int(start_m)+60)
        start_h = str(int(h)-1)
    if end_m > "59":
        end_m = str(int(end_m)-60)
        end_h = str(int(h)+1)

    _start = ''.join((full_time[:8], start_h, full_time[10:]))
    start = ''.join((_start[:10], start_m, _start[12:]))
    _end = ''.join((full_time[:8], end_h, full_time[10:]))
    end = ''.join((_end[:10], end_m, _end[12:]))

    return start, end

def set_phase(row, trn_idx, dev_idx, tst_idx):
    if row['label_idx'] in trn_idx:
        phase = 'train'
    elif row['label_idx'] in dev_idx:
        phase = 'dev'
    elif row['label_idx'] in tst_idx:
        phase = 'test'
    return phase

if __name__=='__main__':
    # set radar_path and load WaveParam_2020.csv
    radar_path = '/media/lepoeme20/Data/projects/daewoo/brave/waveradar/WaveParam_2020.csv'
    # set data_path
    data_path = '/media/lepoeme20/Data/projects/daewoo/brave/crop'

    radar_df = pd.read_csv(radar_path, index_col=None)
    radar_df = radar_df.rename(columns={"Date&Time": "Date"})

    # set folder (date)
    folders = sorted(os.listdir(data_path))

    total_img, total_time, total_idx = [], [], []
    total_direction, total_height, total_period = [], [], []
    label_idx = 0
    for folder in folders:
        print(folder)
        # extract specific time and empty rows
        df = radar_df[radar_df.Date.str.contains(folder[:10], case=False)]
        df = df[df.Date.str[11:13] > '06']
        df = df[df.Date.str[11:13] < '17']
        radar = df[df[' SNR'] != 0.]

        # get images
        all_imgs = sorted(os.listdir(os.path.join(data_path, folder)))
        _imgs = list(filter(lambda x: 7 <= int(x[8:10]) < 17, all_imgs))

        # create empty lists for append
        height_list, direction_list, period_list = [], [], []
        img_list, time_list, idx_list = [], [], []

        for idx in range(radar.shape[0]):
            time = radar['Date'].iloc[idx]
            height = radar[' T.Hs'].iloc[idx]
            direction = radar[' T.Dp'].iloc[idx]
            period = radar[' T.Tp'].iloc[idx]
            # get proper images
            start, end = cal_time(time)
            imgs = list(filter(lambda x: start <= x[:-6] <= end, _imgs))
            # save images
            image_path = [os.path.join(data_path, folder, impath) for impath in imgs]
            img_list.extend(image_path)
            # save label
            height_list.extend([height]*len(imgs))
            direction_list.extend([direction]*len(imgs))
            period_list.extend([period]*len(imgs))
            # save time
            time_list.extend([time]*len(imgs))
            # save label index for i.i.d. condition sampling
            idx_list.extend([label_idx]*len(imgs))
            label_idx += 1

        # append data
        total_img.extend(img_list)
        total_time.extend(time_list)
        total_height.extend(height_list)
        total_direction.extend(direction_list)
        total_period.extend(period_list)
        total_idx.extend(idx_list)

    # create dictionary for build pandas dataframe
    data_dict = {
        'time':total_time,
        'image':total_img,
        'height':total_height,
        'direction': total_direction,
        'period': total_period,
        'label_idx': total_idx
    }
    df = pd.DataFrame(data_dict)

    np.random.seed(22)

    # Time Series
    unique_id = np.unique(df['label_idx'])
    trn_idx, dev_idx, tst_idx = np.split(
        unique_id, [int(.6*len(unique_id)), int(.8*len(unique_id))]
        )
    df['time_phase'] = df.apply(lambda row: set_phase(row, trn_idx, dev_idx, tst_idx), axis=1)

    # i.i.d condition
    np.random.shuffle(unique_id)
    trn_idx, dev_idx, tst_idx = np.split(
        unique_id, [int(.6*len(unique_id)), int(.8*len(unique_id))]
        )
    df['iid_phase'] = df.apply(lambda row: set_phase(row, trn_idx, dev_idx, tst_idx), axis=1)

    df.to_csv('./brave_data_label.csv', index=False)