import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ae_regressor.model_ae import AE, CAE

def create_model(args):
    if args.cae:
        autoencoder = CAE()
        print("CAE will be used")
    else:
        autoencoder = AE()
        print("Linear AE will be used")
    if torch.cuda.is_available():
        autoencoder = nn.DataParallel(autoencoder)

    autoencoder.to(args.device)
    print(f"Model moved to {args.device}")
    return autoencoder


def get_original_data(args, data, phase):
    root_path = f'./ae_regressor/data/{args.label_type}/norm_{args.norm_type}/{args.data_type}/img_flatten/original'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'{phase}_seed_{args.seed}.pkl')
    print(data_path)

    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        total_x, y = data['x'], data['y']
    else:
        img_path = data['image'].values
        if args.label_type == 'height':
            y = data['height'].values
        elif args.label_type == 'direction':
            y= data['direction'].values
        else:
            y = data['period'].values
        # x = np.empty([0, 32*32])
        total_x = np.empty([len(img_path), 32*32])

        for i in range(len(img_path)):
            frame = cv2.imread(img_path[i])
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_AREA)

            # make a 1-dimensional view of arr
            flat_arr = frame.ravel().reshape(1, -1)
            # x = np.r_[x, flat_arr]
            total_x[i] = flat_arr
            if i%1000 == 0:
                print(f'Progress: [{i}/{len(img_path)}]')

        print("save data")
        data = {'x': total_x, 'y': y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return total_x, y

def get_data(args, data_loader, model, phase):
    print(f"Latent vectors will be extracted on {args.device}")
    root_path = f'./ae_regressor/data/{args.label_type}/norm_{args.norm_type}/{args.data_type}/{args.ae_type}'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'{phase}_seed_{args.seed}.pkl')

    if os.path.isfile(data_path):
        print("Load data")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        total_x, total_y = data['x'], data['y']
    else:
        print("build data")
        total_x = np.empty([len(data_loader), 64])
        total_y = np.empty([len(data_loader)])

        for i, (inputs, labels) in enumerate((data_loader)):
            encoded = model(build_input(args, inputs))
            if args.cae:
                latent_vector = torch.squeeze(_gap(encoded)).cpu().data.numpy()
            else:
                latent_vector = encoded.cpu().data.numpy()
            total_x[i] = latent_vector
            total_y[i] = labels.cpu().data.numpy()
            if i%20 == 0:
                print(f'Progress: [{i}/{len(data_loader)}]')

        print("save data")
        data = {'x': total_x, 'y': total_y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return total_x, total_y

def build_input(args, inputs):
    return inputs.to(args.device) if args.cae else inputs.view(inputs.size(0), -1).to(args.device)

def _gap(inputs):
    return nn.AdaptiveAvgPool2d((1, 1))(inputs)

def get_cls_label(labels, dataset):
    q1, q2, q3 = None, None, None
    if dataset == 'brave':
        q1, q2, q3 = 1.17, 1.3, 2.0
    elif dataset == 'weather':
        q1, q2, q3 = 0.591, 0.713, 0.920
    labels[labels < q1] = 0
    labels[labels >= q1] = 1
    labels[labels >= q2] = 2
    labels[labels >= q3] = 3

    return labels.type(torch.LongTensor)