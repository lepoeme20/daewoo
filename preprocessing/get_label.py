import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='data name'
    )
    parser.add_argument(
        '--radar-path', type=str, help='radar csv path'
    )
    parser.add_argument(
        '--data-path', type=str, help='original data path'
    )
    parser.add_argument(
        '--num-worker', type=int, default=4, help='# of workers for Pool'
    )
    params, _ = parser.parse_known_args()
    return params

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
    phase = None
    if row['label_idx'] in trn_idx:
        phase = 'train'
    elif row['label_idx'] in dev_idx:
        phase = 'dev'
    elif row['label_idx'] in tst_idx:
        phase = 'test'
    return phase

def set_date_format(date):
    split_date = date.split('_')
    new_date = f'{split_date[0]}-{split_date[1]}-{split_date[2]} {split_date[3]}:{split_date[4]}'
    return new_date


if __name__=='__main__':
    args = get_args()

    total_img, total_time, total_idx = [], [], []
    total_direction, total_height, total_period = [], [], []

    if args.dataset == 'brave':
        # set radar_path and load WaveParam_2020.csv
        # radar_path = '/media/lepoeme20/Data/projects/daewoo/brave/waveradar/WaveParam_2020.csv'
        # set data_path
        # data_path = '/media/lepoeme20/Data/projects/daewoo/brave/crop'

        radar_df = pd.read_csv(args.radar_path, index_col=None)
        radar_df = radar_df.rename(columns={"Date&Time": "Date"})

        # set folder (date)
        folders = sorted(os.listdir(args.data_path))

        label_idx = 0
        for folder in folders:
            print(folder)
            # extract specific time and empty rows
            df = radar_df[radar_df.Date.str.contains(folder[:10], case=False)]
            df = df[df.Date.str[11:13] > '06']
            df = df[df.Date.str[11:13] < '17']
            radar = df[df[' SNR'] != 0.]

            # get images
            all_imgs = sorted(os.listdir(os.path.join(args.data_path, folder)))
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
                image_path = [os.path.join(args.data_path, folder, impath) for impath in imgs]
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

    elif args.dataset == 'weather_4':
        # radar_path = '/media/lepoeme20/Data/projects/daewoo/weather/wavex_11.csv'
        # set data_path
        # data_path = '/media/lepoeme20/Data/projects/daewoo/weather/data'

        radar_df = pd.read_csv(args.radar_path, index_col=None)
        # get code name
        columns = radar_df.iloc[9, :].values
        # change first column name to 'Date'
        columns[0] = 'Date'
        # rename columns
        radar_df.columns = columns
        # remove rows except data
        radar_df = radar_df.iloc[13: , :]
        # change date column type to datetime
        radar_df.Date = pd.to_datetime(radar_df.Date, format="%y-%m-%d %H:%M %p")

        # set folder (date)
        folders = sorted(os.listdir(args.data_path))
        radar_df.set_index('Date', inplace=True)

        label_idx = 0
        for folder in folders:
            if os.path.isdir(os.path.join(args.data_path, folder)):
                print(folder)
                df = radar_df[folder[:10]]

                # get images
                all_imgs = sorted(os.listdir(os.path.join(args.data_path, folder)))

                # create empty lists for append
                height_list, direction_list, period_list = [], [], []
                img_list, time_list, idx_list = [], [], []

                for idx in range(radar_df.shape[0]):
                    # time = datetime.strftime(radar_df.index[0], "%Y%m%d%H%M%S")
                    time = radar_df.index[idx]
                    height = float(radar_df['Hm0'].iloc[idx])
                    period = float(radar_df['Tm02'].iloc[idx])
                    direction = float(radar_df['Dp1-t'].iloc[idx])

                    # set time
                    # local time = UTC time + 9 hours
                    # 기상 1호는 할 필요 없음 
                    # aligned_time = time + timedelta(hours=9)
                    # start = datetime.strftime((aligned_time - timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    # end = datetime.strftime((aligned_time + timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    start = datetime.strftime((time - timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    end = datetime.strftime((time + timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    imgs = list(filter(lambda x: start <= x[:-6] <= end, all_imgs))

                    # save images
                    image_path = [os.path.join(args.data_path, folder, impath) for impath in imgs]
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

    elif args.dataset == 'weather_1':
        # radar_path = '/media/lepoeme20/Data/projects/daewoo/weather/wavex_11.csv'
        # set data_path
        # data_path = '/media/lepoeme20/Data/projects/daewoo/weather/data'

        radar_df = pd.read_csv(args.radar_path, index_col=None)
        radar_df.DATE = [set_date_format(date) for date in radar_df.DATE.values]
        # change date column type to datetime
        radar_df.DATE = pd.to_datetime(radar_df.DATE, format="%Y-%m-%d %H:%M")

        # set folder (date)
        folders = sorted(os.listdir(args.data_path))
        radar_df.set_index('DATE', inplace=True)

        label_idx = 0
        for folder in folders:
            if os.path.isdir(os.path.join(args.data_path, folder)):
                print(folder)
                df = radar_df[folder]

                # get images
                all_imgs = sorted(os.listdir(os.path.join(args.data_path, folder)))

                # create empty lists for append
                height_list, direction_list, period_list = [], [], []
                img_list, time_list, idx_list = [], [], []

                for idx in range(radar_df.shape[0]):
                    # time = datetime.strftime(radar_df.index[0], "%Y%m%d%H%M%S")
                    time = radar_df.index[idx]
                    height = float(radar_df['SWH'].iloc[idx])
                    period = float(radar_df['SWT'].iloc[idx])
                    direction = float(radar_df['DIR'].iloc[idx])

                    # set time
                    # local time = UTC time + 9 hours
                    # 기상 1호는 할 필요 없음 
                    # aligned_time = time + timedelta(hours=9)
                    # start = datetime.strftime((aligned_time - timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    # end = datetime.strftime((aligned_time + timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    start = datetime.strftime((time - timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    end = datetime.strftime((time + timedelta(minutes=2, seconds=30)), "%Y%m%d%H%M%S")
                    imgs = list(filter(lambda x: start <= x[:-6] <= end, all_imgs))

                    # save images
                    image_path = [os.path.join(args.data_path, folder, impath) for impath in imgs]
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
    for i in range(5):
        np.random.shuffle(unique_id)
        trn_idx, dev_idx, tst_idx = np.split(
            unique_id, [int(.6*len(unique_id)), int(.8*len(unique_id))]
            )
        print(f'iid_phase{i}')
        df[f'iid_phase_{i}'] = df.apply(lambda row: set_phase(row, trn_idx, dev_idx, tst_idx), axis=1)

    df.to_csv(f'./{args.dataset}_data_label.csv', index=False)
