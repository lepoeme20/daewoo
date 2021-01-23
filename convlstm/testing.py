import os
import argparse
import pandas as pd
import numpy as np
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
    _, _, tst_loader = get_dataloader(args.csv_path, args.root_img_path, args.batch_size)
    
    model = ConvLSTMModel(args.mem_size)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    trues = np.array([])
    preds = np.array([])

    print('==> Start Testing')
    model.eval()
    with torch.no_grad():
        for inputs, targets in tst_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)

            trues = np.r_[trues, targets.detach().cpu().numpy()]
            preds = np.r_[preds, outputs.detach().cpu().numpy().squeeze(1)]
        
        mse, mae, mape = return_perf(trues, preds)

        print('Test MSE: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(mse, mae, mape))

    result = pd.DataFrame()
    result['true'] = list(trues)
    result['pred'] = list(preds)

    os.makedirs(args.save_root_path, exist_ok=True)
    result.to_csv(os.path.join(args.save_root_path, 'result.csv'), index=False)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--mem-size', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--model-path', type=str, default='./result/best_model.pt')
    parser.add_argument('--save-root-path', type=str, default='./result/')
    parser.add_argument('--csv-path', type=str, default='/media/heejeong/HDD2/project/daewoo/data/weather_4/wave_radar/weather_4_data_label.csv',
                        help='Directory csv file of labels')
    parser.add_argument('--root-img-path', type=str, default='/media/heejeong/HDD2/project/daewoo/data/weather_4/data_crop/',
                        help='Directory folder of images')
    args, _ = parser.parse_known_args()

    main(args)
