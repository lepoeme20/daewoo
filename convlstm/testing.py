import os
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch

from model import ConvLSTMModel
from build_dataloader import get_dataloader


def return_perf(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, mape

def main(args):
    # Log files
    if args.use_total_phase:
        folder_name = 'result_source_total_phase'
        folder_name += '_split_{}'.format(args.img_split_type)
        source_data = 'total_phase'
    else:
        if args.use_time_phase:
            folder_name = 'result_source_time_phase'
            folder_name += '_split_{}'.format(args.img_split_type)
            source_data = 'time_phase'
        else:
            folder_name = 'result_source_'
            if args.use_weather_1:
                folder_name += '1'
            if args.use_weather_4:
                folder_name += '4'
            folder_name += '_split_{}_seed_{}'.format(args.img_split_type, args.iid)
            source_data = folder_name.split('source_')[-1].split('_split')[0]
    
    model = ConvLSTMModel(args.mem_size, args.img_split_type)
    model.load_state_dict(torch.load(os.path.join(args.save_root_path, folder_name, 'best_model.pt')))
    model.cuda()

    # if args.use_total_phase:
    #     _, _, tst_loader_14_label = get_dataloader(False, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                                False, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                                True, args.weather_total_phase_csv_path,
    #                                                False, args.img_split_type, args.iid, args.batch_size)
        
    #     data_loaders = {'total_phase': tst_loader_14_label}
    #     results = {'total_phase': {'trues': np.array([]), 'preds': np.array([])}}
    # else:
    #     if args.use_time_phase:
    #         _, _, tst_loader_time = get_dataloader(args.use_weather_1, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                                args.use_weather_4, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                                False, args.weather_total_phase_csv_path,
    #                                                True, args.img_split_type, args.iid, args.batch_size)
        
    #         data_loaders = {'time_phase': tst_loader_time}
    #         results = {'time_phase': {'trues': np.array([]), 'preds': np.array([])}}
    #     else:
    #         _, _, tst_loader_1 = get_dataloader(True, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                             False, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                             False, args.weather_total_phase_csv_path,
    #                                             False, args.img_split_type, args.iid, args.batch_size)
    #         # 기상 1호 4차 데이터 로더
    #         _, _, tst_loader_4 = get_dataloader(False, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                             True, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                             False, args.weather_total_phase_csv_path,
    #                                             False, args.img_split_type, args.iid, args.batch_size)
    #         # 기상 1호 1차 & 4차 데이터 로더
    #         _, _, tst_loader_14 = get_dataloader(True, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                             True, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                             False, args.weather_total_phase_csv_path,
    #                                             False, args.img_split_type, args.iid, args.batch_size)

    #         data_loaders = {'weather_1': tst_loader_1, 'weather_4': tst_loader_4, 'weather_14': tst_loader_14}
    #         results = {'weather_1': {'trues': np.array([]), 'preds': np.array([])},
    #                    'weather_4': {'trues': np.array([]), 'preds': np.array([])},
    #                    'weather_14': {'trues': np.array([]), 'preds': np.array([])}}

    _, _, tst_loader = get_dataloader(
        args.brave_csv_path, args.brave_root_img_path, args.use_time_phase,
        args.img_split_type, args.batch_size
    )
    data_loaders = {'new_brave': tst_loader}
    results = {
        'new_brave': {'trues': np.array([]), 'preds': np.array([])}
    }

    print('==> Start Testing')
    for target_data in data_loaders:
        model.eval()
        with torch.no_grad():
            for inputs, targets in data_loaders[target_data]:
                inputs = inputs.cuda()
                outputs = model(inputs)

                results[target_data]['trues'] = np.r_[results[target_data]['trues'], targets.numpy()]
                results[target_data]['preds'] = np.r_[results[target_data]['preds'], outputs.detach().cpu().numpy().squeeze(1)]
            
            mse, mae, mape = return_perf(results[target_data]['trues'], results[target_data]['preds'])
            print('Source: weather_{} & Target: {}'.format(source_data, target_data))
            print('Test MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}\n'.format(mse, mae, mape))

        result = pd.DataFrame()
        result['true'] = list(results[target_data]['trues'])
        result['pred'] = list(results[target_data]['preds'])

        if args.use_total_phase:
            file_name = 'result_source_target_total_phase.csv'
        else:
            file_name = 'result_source_{}_target_{}.csv'.format(source_data, target_data.split('_')[-1])
        result.to_csv(os.path.join(args.save_root_path, folder_name, file_name), index=False)

def baseline(args):
    device = torch.device('cuda')
    # data loader
    trn_loader, _, tst_loader = get_dataloader(
        args.brave_csv_path, args.brave_root_img_path, args.use_time_phase,
        args.img_split_type, args.batch_size
    )

    with torch.no_grad():
        # trn_mean = torch.zeros((len(trn_loader), 1), device=device)
        # for idx, (_, label) in enumerate(trn_loader):
        #     batch_height = label.to(device)
        #     trn_mean[idx] = torch.mean(batch_height).item()

        # trn_mean = torch.mean(trn_mean)
        # with open('new_brave_trn_mean.p', 'wb') as f:
        #     pickle.dump(trn_mean, f)
        with open('new_brave_trn_mean.p', 'rb') as f:
            trn_mean = pickle.load(f)
        mae = 0.0
        mape = 0.0
        for step, (_, label) in enumerate(tst_loader):
            y_tst = label.to(device)
            mean_value = torch.full_like(y_tst, trn_mean)
            mae += MAE(mean_value, y_tst).item()
            mape += MAPE(mean_value, y_tst).item()

        mae /= step + 1
        mape /= step + 1

        print(mae, mape)
    return mae, mape

def MAE(pred, true):
    return torch.mean((pred - true).abs())

def MAPE(pred, true):
    return torch.mean((pred - true).abs() / (true.abs() + 1e-8)) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mem-size', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--save-root-path', type=str, default='./result/')
    # 변경해야할 옵션
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--img_split_type', type=int, default=0,
                        help='0: img to 3 frames vertically | 1: img to 5 frames vertically | 2: img to 3 frames horizontally')
    parser.add_argument('--use-weather-1', type=bool, default=False)
    parser.add_argument('--use-weather-4', type=bool, default=False)
    parser.add_argument('--use-brave', type=bool, default=True)
    parser.add_argument('--use-total-phase', type=bool, default=False)
    parser.add_argument('--use-time-phase', type=bool, default=False)
    parser.add_argument('--weather-1-csv-path', type=str,
                        default='/media/heejeong/HDD2/project/daewoo/data/weather_1/wave_radar/weather_1_data_label_seed.csv',
                        help='Csv file directory of labels for weather1')
    parser.add_argument('--weather-1-root-img-path', type=str,
                        default='/media/heejeong/HDD2/project/daewoo/data/weather_1/data_crop/',
                        help='Csv file directory of labels for weather1')
    parser.add_argument('--weather-4-csv-path', type=str,
                        default='/media/heejeong/HDD2/project/daewoo/data/weather_4/wave_radar/weather_4_data_label_seed.csv',
                        help='Csv file directory of labels for weather4')
    parser.add_argument('--weather-4-root-img-path', type=str,
                        default='/media/heejeong/HDD2/project/daewoo/data/weather_4/data_crop/',
                        help='Folder directory of images for weather4')
    parser.add_argument('--weather-total-phase-csv-path', type=str,
                        default='/media/heejeong/HDD2/project/daewoo/data/weather_1_4_data_label_seed.csv',
                        help='Csv file directory of labels for weather1&4')
    parser.add_argument('--brave-csv-path', type=str,
                        default='../preprocessing/brave_data_label.csv')
    parser.add_argument('--brave-root-img-path', type=str,
                        default='/media/lepoeme20/Data/projects/daewoo/hyundai_brave/crop_data')
    args, _ = parser.parse_known_args()

    # main(args)
    baseline(args)
