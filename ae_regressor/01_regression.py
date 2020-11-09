import os
import sys
import torch
# Regressor
from sklearn.svm import SVR
from sklearn.utils import parallel_backend
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
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

def main(args):
    if args.use_original:
        df = pd.read_csv(args.csv_path)

        if args.iid:
            # i.i.d condition
            trn, _, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        else:
            # time series condition
            trn, _, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])

    else:
        trn_loader, _, tst_loader = get_dataloader(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            iid=args.iid,
            transform=args.norm_type,
            img_size=args.img_size
        )

    # Set & create save path
    if args.iid:
        print("** Training progress with iid condition **")
        model_path = f'./ae_regressor/best_model/norm_{args.norm_type}/iid'
    else:
        print("** Training progress with time series condition **")
        model_path = f'./ae_regressor/best_model/norm_{args.norm_type}/time'

    # Create model
    autoencoder = F.create_model(args)
    checkpoint = torch.load(os.path.join(model_path, 'autoencoder.pkl'))
    autoencoder.module.load_state_dict(checkpoint['model'])
    encoder = autoencoder.module.encoder

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

        load_path = os.path.join(model_path, 'regression.pkl')
        with open(load_path, 'rb') as f:
            best_model = pickle.load(f)

        x_test, y_test = F.get_data(tst_loader, args.device, encoder)
        y_pred = best_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
        print(f"[Test] MAE:{mae}, MAPE:{mape}")

    else:
        # train data 추출
        if args.use_original:
            x_train, y_train = F.get_original_data(trn, args.sampling_ratio)
        else:
            x_train, y_train = F.get_data(trn_loader, args.device, encoder)

        # gird search 범위 지정
        bound = [0.001, 0.01, 0.1, 1., 10, 100]
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': bound, # Regularization parameter
            'gamma': bound
            }
        # grid_search 지정
        grid_search = GridSearchCV(
            estimator=SVR(),
            param_grid=param_grid,
            cv=5,
            n_jobs=args.num_parallel,
            scoring=make_scorer(mean_absolute_error),
            return_train_score=True,
            verbose=10)
        # model fitting
        with parallel_backend('multiprocessing'):
            grid_search.fit(x_train, y_train)

        # printing
        print(grid_search.best_params_)
        print(grid_search.best_estimator_)

        # set best model
        best_model = grid_search.best_estimator_

        # fit
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_train)
        mae = mean_absolute_error(y_train, y_pred)
        mape = mean_absolute_percentage_error(y_true=y_train, y_pred=y_pred)
        print(f"[Trainig] MAE:{mae}, MAPE:{mape}")

        performance = {'mae': mae, 'mape': mape}
        with open(os.path.join(model_path, 'dev_performance.pkl'), 'wb') as f:
            pickle.dump(performance, f)

        save_path = os.path.join(model_path, 'regression.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(best_model, f)


if __name__ == '__main__':
    # Set random seed for reproducibility
    SEED = 87
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    h_params = config.get_config()
    main(h_params)