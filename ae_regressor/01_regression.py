import os
import sys
import torch
# Regressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle
# import matplotlib.pyplot as plt
# TODO: 시각화 추가
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader
import utils.functions as F
from ae_regressor import config


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    if args.use_original:
        df = pd.read_csv(args.csv_path)

        if args.iid:
            trn, dev, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        else:
            trn, dev, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])
        model_path = f'./ae_regressor/best_model/{args.label_type}/norm_{args.norm_type}/{args.data_type}/img_flatten/original'

    else:
        trn_loader, dev_loader, tst_loader = get_dataloader(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            dtype=args.data_type,
            iid=args.iid,
            transform=args.norm_type,
            img_size=args.img_size
        )

        # Create model
        model_path = f'./ae_regressor/best_model/{args.label_type}/norm_{args.norm_type}/{args.data_type}/CAE'
        autoencoder = F.create_model(args)
        checkpoint = torch.load(os.path.join(model_path, 'autoencoder.pkl'))
        autoencoder.module.load_state_dict(checkpoint['model'])
        encoder = autoencoder.module.encoder

    os.makedirs(model_path, exist_ok=True)
    print(f"** Training progress with {args.data_type} condition **")

    if args.test:
        # #reg_type = 'svr'

        # total_labels = []
        # total_plabels = []

        # for idx, (inputs, labels) in enumerate((tst_loader)):
        #     # step progress #
        #     inputs = get_torch_vars(inputs)
        #     encoded, _ = AE(inputs)

        #     regressor = Regressor(encoded, labels, train=False, reg_type='svr', iid_type = 'time')
        #     predict_label = regressor(encoded, labels, train=False, reg_type='svr', iid_type = 'time')

        #     labels = labels.numpy()
        #     total_labels.extend(labels)
        #     total_plabels.extend(predict_label)
        #     print(predict_label)

        # x = [i for i in range(len(total_labels))]

        # plt.plot(x, total_labels)
        # plt.plot(x, total_plabels)
        # plt.legend(['Actual Label','Predicted Label'])
        # plt.title('Data Type: {}, Regressor: {}'.format(regressor.iid_type, regressor.reg_type))
        # plt.savefig('{}_{}.png'.format(regressor.iid_type, regressor.reg_type))
        # plt.show()
        # exit(0)
        load_path = os.path.join(model_path, 'lightgbm.pkl')
        with open(load_path, 'rb') as f:
            best_model = pickle.load(f)

        if args.use_original:
            x_tst, y_tst = F.get_original_data(args, tst, 'tst')
        else:
            x_tst, y_tst = F.get_data(args, trn_loader, encoder, 'tst')

        d_test = lgb.Dataset(data=x_tst, label = y_tst)
        y_pred = best_model.predict(d_test)
        mae = mean_absolute_error(y_tst, y_pred)
        mape = mean_absolute_percentage_error(y_true=y_tst, y_pred=y_pred)
        print(f"[Test] MAE:{mae}, MAPE:{mape}")

    else:
        # train data 추출
        if args.use_original:
            x_train, y_train = F.get_original_data(args, trn, 'trn')
            x_dev, y_dev = F.get_original_data(args, dev, 'dev')
        else:
            x_train, y_train = F.get_data(args, trn_loader, encoder, 'trn')
            x_dev, y_dev = F.get_data(args, dev_loader, encoder, 'dev')
        print(f'Data volumn for grid search: {len(y_train)}')

        d_train = lgb.Dataset(data=x_train, label = y_train)
        d_dev = lgb.Dataset(data=x_dev, label=y_dev)

        params = {}
        params['learning_rate'] = 0.1
        params['boosting_type'] = 'gbdt'
        params['objective'] = 'regression_l1'
        params['metric'] = 'mae'
        params['num_leaves'] = 32 # defualt: 31
        params['min_data'] = 20 # number of data in a leaf: overfitting, default: 20
        params['device'] = 'cpu'
        params['bagging_fraction'] = 0.3
        params['bagging_freq'] = 10
        params['lambda_l1'] = 0.7

        model = lgb.train(
            params=params,
            train_set=d_train,
            num_boost_round=2000,
            valid_sets=d_dev,
            verbose_eval=100,
            early_stopping_rounds=100
            )
        predict_dev = model.predict(x_dev)
        mae = mean_absolute_error(y_dev, predict_dev)
        mape = mean_absolute_percentage_error(y_dev, predict_dev)

        print(f"MAE: {mae}, MAPE: {mape}")

        with open(os.path.join(model_path, 'lightgbm.pkl'), 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    global args
    # Set random seed for reproducibility
    args = config.get_config()
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    main()