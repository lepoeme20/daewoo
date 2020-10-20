# Torch
import torch
import torch.nn as nn

# Module
from train_AE import create_model, get_torch_vars
from utils.build_dataloader import get_dataloader

# Regressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression as LR
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

def mean_absolute_percentage_error(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Regressor(nn.Module):
    def __init__(self, x, labels,train, reg_type, iid_type):
        super(Regressor, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1))
        self.svr = SVR(kernel= 'rbf', C=100, gamma=0.1, epsilon=0.1)
        self.lr = LR()
        self.reg_type = reg_type
        self.iid_type = iid_type

    def forward(self, x, labels, train, reg_type, iid_type):
        x = self.gap(x)
        x = x.view(-1, 48).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        if train == True:
            if reg_type == 'svr':
                model = self.svr.fit(x, labels)
                file_path = './model/regressor_SVR_' + iid_type + '.pkl'
                print(file_path)
            if reg_type == 'lr':
                model = self.lr.fit(x, labels)
                file_path = './model/regressor_LR_' + iid_type + '.pkl'

            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            output = model.predict(x)

        else: # train == False
            if reg_type == 'svr':
                file_path = './model/regressor_SVR_' + iid_type + '.pkl'
            if reg_type == 'lr':
                file_path = './model/regressor_LR_' + iid_type + '.pkl'
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            output = model.predict(x)

        return output


def main_reg():
    parser = argparse.ArgumentParser(description="Visualize Regression Result")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="To Plot Predicted Result.")
    args = parser.parse_args()

    AE = create_model()

    # if iid = False:
    AE.load_state_dict(torch.load("./autoencoder_daewoo_time.pkl"))
    # if iid = True:
    # AE.load_state_dict(torch.load("./autoencoder_daewoo.pkl"))

    csv_path = './preprocessing/brave_data_label.csv'
    _, dev_loader, tst_loader = get_dataloader(csv_path, iid=False)


    if args.visualize:
        #reg_type = 'svr'

        total_labels = []
        total_plabels = []

        for idx, (inputs, labels) in enumerate((tst_loader)):
            # step progress #
            inputs = get_torch_vars(inputs)
            encoded, _ = AE(inputs)

            regressor = Regressor(encoded, labels, train=False, reg_type='svr', iid_type = 'time')
            predict_label = regressor(encoded, labels, train=False, reg_type='svr', iid_type = 'time')

            labels = labels.numpy()
            total_labels.extend(labels)
            total_plabels.extend(predict_label)
            print(predict_label)

        x = [i for i in range(len(total_labels))]

        plt.plot(x, total_labels)
        plt.plot(x, total_plabels)
        plt.legend(['Actual Label','Predicted Label'])
        plt.title('Data Type: {}, Regressor: {}'.format(regressor.iid_type, regressor.reg_type))
        plt.savefig('{}_{}.png'.format(regressor.iid_type, regressor.reg_type))
        plt.show()
        exit(0)


    total_train_loss = 0.0
    for i, (inputs, labels) in enumerate((dev_loader)):
        inputs = get_torch_vars(inputs)
        encoded, _ = AE(inputs) # shape of latent vector: torch.Size([32, 48, 17, 17])
        train_loss = 0.0

        for epoch in range(10):
            regressor = Regressor(encoded, labels,train=True, reg_type = 'svr', iid_type = 'time')
            predict_label = regressor(encoded, labels,train=True, reg_type = 'svr', iid_type = 'time')

            mse = mean_squared_error(predict_label, labels)
            train_loss += mse
        train_loss /= 10
        total_train_loss += train_loss
        print('MSE: %.5f' % (train_loss))
    print('Average MSE: %.5f' % (total_train_loss / len(dev_loader)))

    # for test
    test_loss = 0.0
    test_loss2 = 0.0

    for idx, (inputs, labels) in enumerate((tst_loader)):
        # step progress #
        inputs = get_torch_vars(inputs)
        encoded, _ = AE(inputs)

        regressor = Regressor(encoded, labels, train=False, reg_type = 'svr', iid_type = 'time')
        predict_label = regressor(encoded, labels, train=False, reg_type = 'svr', iid_type = 'time')

        mae = mean_absolute_error(predict_label, labels)
        mape = mean_absolute_percentage_error(predict_label, labels.numpy())
        test_loss += mae
        test_loss2 += mape
        print(idx, len(tst_loader), 'MAE: %.5f, ' % ((test_loss / (idx + 1))))
        print(idx, len(tst_loader), 'MAPE: %.5f, ' % ((test_loss2 / (idx + 1))))


if __name__ == '__main__':
    main_reg()