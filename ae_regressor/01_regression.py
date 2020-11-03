import os
import sys
import torch
import torch.nn as nn
# Regressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import GridSearchCV
import pickle
# import matplotlib.pyplot as plt
# TODO: 시각화 추가
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader
from ae_regressor.model_ae import AE

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_model(args):
    autoencoder = AE()
    if torch.cuda.is_available():
        autoencoder = nn.DataParallel(autoencoder)

    autoencoder.to(args.device)
    print(f"Model moved to {args.device}")
    return autoencoder

def get_data(data_loader, device, model):
    print(f"Latent vectors will be extracted on {device}")
    x = np.empty([0, 32])
    y = np.empty([0])
    for i, (inputs, labels) in enumerate((data_loader)):
        encoded = model(inputs.view(inputs.size(0), -1).to(device))
        latent_vector = encoded.cpu().data.numpy()
        x = np.r_[x, latent_vector]
        y = np.r_[y, labels.cpu().data.numpy()]
        if i%20 == 0:
            print(f'Progress: [{i}/{len(data_loader)}]')
    # inputs, labels = next(iter(data_loader))
    # encoded = model(inputs.view(inputs.size(0), -1).to(device))
    # latent_vector = encoded.cpu().data.numpy()
    # x = np.r_[x, latent_vector]
    # y = np.r_[y, labels.cpu().data.numpy()]
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument(
        "--csv_path", type=str, default='./preprocessing/brave_data_label.csv',
        help="csv file path"
    )
    parser.add_argument(
        "--iid", action="store_true", default=False, help="use argument for iid condition"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Perform Test only"
    )
    parser.add_argument(
        "--norm-type", type=int, choices=[0, 1, 2],
        help="0: ToTensor, 1: Ordinary image normalization, 2: Image by Image normalization"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=32, help='image size for Auto-encoder (default: 32x32)'
    )
    args, _ = parser.parse_known_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    autoencoder = create_model(args)
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

        x_test, y_test = get_data(tst_loader, args.device, encoder)
        y_pred = best_model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
        print(f"[Test] MAE:{mae}, MAPE:{mape}")

    else:
        # train data 추출
        x_train, y_train = get_data(trn_loader, args.device, encoder)

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
            n_jobs=64,
            scoring=make_scorer(mean_absolute_error),
            return_train_score=True,
            verbose=10)
        # model fitting
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
        with open(os.path.join(model_path, 'dev_performance.pkl')) as f:
            pickle.dump(performance, f)

        save_path = os.path.join(model_path, 'regression.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(best_model, f)


if __name__ == '__main__':
    main()