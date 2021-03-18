import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import ConvLSTMModel
from update_build_dataloader import get_dataloader


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
    
    save_path = os.path.join(args.save_root_path, folder_name, f'time_{args.use_time_phase}')
    os.makedirs(save_path, exist_ok=True)
    inference_result = open(os.path.join(save_path, 'inference_result.txt'), 'w')
    update_log_loss = open(os.path.join(save_path, 'update_log_loss.txt'), 'w')


    # -------------------------------
    # LOAD PRETRAINED MODEL
    # -------------------------------
    model = ConvLSTMModel(args.mem_size, args.img_split_type)

    for i in range(1,6):
        model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
        model.cuda()

        print('[Data Block {}th]'.format(i))
        tst_loader = get_dataloader(
            args.brave_csv_path, args.brave_root_img_path, args.use_time_phase,
            args.img_split_type, args.batch_size, args.use_total_phase, use_prediction = False, group_num = i, save_path = None
        )
        
        # -------------------------------
        # INFERENCE WITH PRETRAINED MODEL
        # -------------------------------
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
                print('Source: {} & Target: {}'.format(source_data, target_data))
                print('Test MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}\n'.format(mse, mae, mape))

                inference_result.write('[{}th data block] Test MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}\n'.format(i-1, mse, mae, mape))


            result = pd.DataFrame()
            result['true'] = list(results[target_data]['trues'])
            result['pred'] = list(results[target_data]['preds'])

            if args.use_total_phase:
                file_name = 'result_source_target_total_phase_beforeupdate{}th.csv'.format(i)
            else:
                file_name = 'result_source_{}_target_{}_beforeupdate{}th.csv'.format(source_data, target_data.split('_')[-1], i)
            result.to_csv(os.path.join(save_path, file_name), index=False)

        # -------------------------------
        # UPDATE MODEL WITH DATA BLOCK
        # -------------------------------
        
        print('==> Start Updating')

        new_loader = get_dataloader(
            args.brave_csv_path, args.brave_root_img_path, args.use_time_phase,
            args.img_split_type, args.batch_size, args.use_total_phase, group_num = i, use_prediction = True, save_path = save_path
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.decay_rate)

        best_loss = 10000
        for epoch in range(args.num_epochs): # EPOCH: 5
            print('==> Start Epoch = {}'.format(epoch + 1))
            epoch_loss = 0
            for i, (inputs, targets) in enumerate(new_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if (i + 1) % 200 == 0:
                    print('Epoch: [{}][{}/{}]\t\t'
                        'Loss: {}\t\t'.format(epoch + 1, i + 1, len(new_loader), loss.item()))
                
            avg_loss = epoch_loss / len(new_loader)
            print('\nUpdating: Loss = {}'.format(avg_loss))

            update_log_loss.write('Updating loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))

            
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
                    

            lr_scheduler.step()
    
    inference_result.close()
    update_log_loss.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='Learning rate decay step')
    parser.add_argument('--decay-rate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--mem-size', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--save-root-path', type=str, default='./result')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--img_split_type', type=int, default=0,
                        help='0: img to 3 frames vertically | 1: img to 5 frames vertically | 2: img to 3 frames horizontally')
    parser.add_argument('--use-weather-1', type=bool, default=False)
    parser.add_argument('--use-weather-4', type=bool, default=False)
    parser.add_argument('--use-brave', type=bool, default=True)
    parser.add_argument('--use-total-phase', type=bool, default=False)
    parser.add_argument('--use-time-phase', type=bool, default=True)
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
                        default='/mnt/C83AFA0C3AF9F6F2/hyundai_brave/crop_data/')
    args, _ = parser.parse_known_args()

    main(args)
