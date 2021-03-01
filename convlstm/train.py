import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn

from model import ConvLSTMModel
from build_dataloader import get_dataloader


def main(args):
    # Seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    # trn_loader, dev_loader, _ = get_dataloader(args.use_weather_1, args.weather_1_csv_path, args.weather_1_root_img_path,
    #                                            args.use_weather_4, args.weather_4_csv_path, args.weather_4_root_img_path,
    #                                            args.use_total_phase, args.weather_total_phase_csv_path,
    #                                            args.use_time_phase, args.img_split_type, args.iid, args.batch_size)
    trn_loader, dev_loader, _ = get_dataloader(
        args.brave_csv_path, args.brave_root_img_path, args.use_time_phase,
        args.img_split_type, args.batch_size
    )
    # Log files
    if args.use_total_phase:
        folder_name = 'result_source_total_phase'
        folder_name += '_split_{}'.format(args.img_split_type)
    else:
        if args.use_time_phase:
            folder_name = 'result_source_time_phase'
            folder_name += '_split_{}'.format(args.img_split_type)
        else:    
            folder_name = 'result_source_'
            if args.use_weather_1:
                folder_name += '1'
            if args.use_weather_4:
                folder_name += '4'
            folder_name += '_split_{}_seed_{}'.format(args.img_split_type, args.iid)

    os.makedirs(os.path.join(args.save_root_path, folder_name), exist_ok=True)
    trn_log_loss = open(os.path.join(args.save_root_path, folder_name, 'trn_log_loss.txt'), 'w')
    dev_log_loss = open(os.path.join(args.save_root_path, folder_name, 'dev_log_loss.txt'), 'w')

    model = ConvLSTMModel(args.mem_size, args.img_split_type)
    model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.decay_rate)

    best_loss = 10000
    for epoch in range(args.num_epochs):
        print('==> Start Epoch = {}'.format(epoch + 1))
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(trn_loader):
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
                      'Loss: {}\t\t'.format(epoch + 1, i + 1, len(trn_loader), loss.item()))
            
        avg_loss = epoch_loss / len(trn_loader)
        print('\nTraining: Loss = {}'.format(avg_loss))

        trn_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch + 1, avg_loss))

        # validation
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for j, (inputs, targets) in enumerate(dev_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))

                val_epoch_loss += loss.item()

            val_avg_loss = val_epoch_loss / len(dev_loader)
            print('Validation: Loss = {}\n'.format(val_avg_loss))

            dev_log_loss.write('Validation Loss after {} epochs = {}\n'.format(epoch + 1, val_avg_loss))
            
            if val_avg_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(args.save_root_path, folder_name, 'best_model.pt'))
                best_loss = val_avg_loss

        lr_scheduler.step()

    trn_log_loss.close()
    dev_log_loss.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='Learning rate decay step')
    parser.add_argument('--decay-rate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--mem-size', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--save-root-path', type=str, default='./result/')
    # 변경해야할 옵션
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
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

    main(args)
