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

def get_results(x, y, top_k, df_results, grid, model_path):
    top_k_parmas = df_results.iloc[top_k, 7]
    print(top_k_parmas)
    top_k_model = grid.best_estimator_.set_params(**top_k_parmas)

    # fit
    top_k_model.fit(x, y)
    y_pred = top_k_model.predict(x)
    mae = mean_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y_true=y, y_pred=y_pred)
    print(f"[Trainig] MAE:{mae}, MAPE:{mape}")

    performance = {'mae': mae, 'mape': mape}
    with open(os.path.join(model_path, f'top_{top_k}_trn_performance.pkl'), 'wb') as f:
        pickle.dump(performance, f)

    save_path = os.path.join(model_path, f'top_{top_k}_regression.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(top_k_model, f)

def set_grid(kernel):
    # gird search 범위 지정
    bound = [0.001, 0.01, 0.1, 1., 10., 100.]
    degrees = [1, 2, 3, 4, 5, 6]
    if kernel == 'poly':
        param_grid = {
            'kernel': ['poly'],
            'C': bound, # Regularization parameter
            'degree': degrees
            }
    else:
        param_grid = {
            'kernel': ['rbf'],
            'C': bound, # Regularization parameter
            'gamma': bound,
            }
    return param_grid

def main(args):
    if args.use_original:
        df = pd.read_csv(args.csv_path)

        if args.iid:
            trn, _, tst = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        else:
            trn, _, tst = np.split(df, [int(.6*len(df)), int(.8*len(df))])
        model_path = f'./ae_regressor/best_model/img_flatten/original/norm_{args.norm_type}/{args.data_type}'

    else:
        trn_loader, _, tst_loader = get_dataloader(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            iid=args.iid,
            transform=args.norm_type,
            img_size=args.img_size
        )

        # Create model
        model_path = f'./ae_regressor/best_model/{args.ae_type}/norm_{args.norm_type}/{args.data_type}'
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

        load_path = os.path.join(model_path, 'regression.pkl')
        with open(load_path, 'rb') as f:
            best_model = pickle.load(f)

        x_test, y_test = F.get_data(args, tst_loader, encoder)
        y_pred = best_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
        print(f"[Test] MAE:{mae}, MAPE:{mape}")

    else:
        # train data 추출
        if args.use_original:
            x_train, y_train = F.get_original_data(args, trn, args.sampling_ratio)
        else:
            x_train, y_train = F.get_data(args, trn_loader, encoder, args.sampling_ratio)
        print(f'Data volumn for grid search: {len(y_train)}')
        results = pd.DataFrame()

        # gird search 범위 지정
        for kernel in ['poly', 'rbf']:
            param_grid = set_grid(kernel)
            # grid_search 지정
            grid_search = GridSearchCV(
                estimator=SVR(),
                param_grid=param_grid,
                cv=3,
                n_jobs=args.num_parallel,
                scoring=make_scorer(mean_absolute_error),
                return_train_score=True,
                )
            # model fitting
            with parallel_backend('multiprocessing'):
                grid_search.fit(x_train, y_train)

            results_tmp = pd.DataFrame(grid_search.cv_results_)
            results = results.append(results_tmp, sort=False)

        # save grid search results
        results.sort_values(by='mean_test_score', inplace=True)
        results.to_csv(os.path.join(model_path, 'gird_search.csv'))

        if args.use_original:
            x_train, y_train = F.get_original_data(args, trn, 1)
        else:
            x_train, y_train = F.get_data(args, trn_loader, encoder, 1)
        for i in range(1, 4):
            get_results(
                x=x_train,
                y=y_train,
                top_k=i,
                df_results=results,
                grid=grid_search,
                model_path=model_path
            )

if __name__ == '__main__':
    # Set random seed for reproducibility
    h_params = config.get_config()
    SEED = h_params.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    h_params = config.get_config()
    main(h_params)