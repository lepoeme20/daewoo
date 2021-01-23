import os
import sys
import glob
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

    trn_loader, dev_loader, _ = get_dataloader(args.csv_path, args.root_img_path, args.batch_size)

    # Log files
    os.makedirs(args.save_root_path, exist_ok=True)
    trn_log_loss = open((args.save_root_path + '/trn_log_loss.txt'), 'w')
    dev_log_loss = open((args.save_root_path + '/dev_log_loss.txt'), 'w')

    model = ConvLSTMModel(args.mem_size)
    model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.decay_rate)

    best_loss = 10000
    for epoch in range(args.num_epochs):
        print('==> Start Epoch = {}'.format(epoch + 1))
        epoch_loss = 0
        train_correct = 0
        for i, (inputs, targets) in enumerate(trn_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 200 == 0:
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
                loss = criterion(outputs, targets)

                val_epoch_loss += loss.item()

            val_avg_loss = val_epoch_loss / len(dev_loader)
            print('Validation: Loss = {}'.format(val_avg_loss))

            dev_log_loss.write('Validation Loss after {} epochs = {}\n'.format(epoch + 1, val_avg_loss))
            
            if val_avg_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(args.save_root_path, 'best_model.pt'))
                best_loss = val_avg_loss

        lr_scheduler.step()

    trn_log_loss.close()
    dev_log_loss.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='Learning rate decay step')
    parser.add_argument('--decay-rate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--mem-size', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--save-root-path', type=str, default='./result/')
    parser.add_argument('--csv-path', type=str, default='/media/heejeong/HDD2/project/daewoo/data/weather_4/wave_radar/weather_4_data_label.csv',
                        help='Directory csv file of labels')
    parser.add_argument('--root-img-path', type=str, default='/media/heejeong/HDD2/project/daewoo/data/weather_4/data_crop/',
                        help='Directory folder of images')
    args, _ = parser.parse_known_args()

    main(args)