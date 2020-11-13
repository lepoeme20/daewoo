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


def get_original_data(args, data, sampling_ratio):
    root_path = f'./ae_regressor/trn_data/img_flatten/original/norm_{args.norm_type}/{args.data_type}'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'sampling_{sampling_ratio}_seed_{args.seed}.pkl')
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        x, y = data['x'], data['y']
    else:
        img_path = data['image'].values
        y = data['label'].values
        x = np.empty([0, 32*32])

        if sampling_ratio != 1.:
            idx = np.random.choice(np.arange(len(y)), int(len(y)*sampling_ratio), replace=False)
            img_path = img_path[idx]
            y = y[idx]

        for i in range(len(img_path)):
            frame = cv2.imread(img_path[i])
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_AREA)

            # make a 1-dimensional view of arr
            flat_arr = frame.ravel().reshape(1, -1)
            x = np.r_[x, flat_arr]
            if i%1000 == 0:
                print(f'Progress: [{i}/{len(img_path)}]')

        data = {'x': x, 'y': y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return x, y

def get_data(args, data_loader, model, sampling_ratio):
    print(f"Latent vectors will be extracted on {args.device}")
    root_path = f'./ae_regressor/trn_data/{args.ae_type}/norm_{args.norm_type}/{args.data_type}'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'sampling_{sampling_ratio}_seed_{args.seed}.pkl')

    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        x, y = data['x'], data['y']
    else:
        x = np.empty([0, 64])
        y = np.empty([0])

        for i, (inputs, labels) in enumerate((data_loader)):
            encoded = model(build_input(args, inputs))
            if args.cae:
                latent_vector = torch.squeeze(_gap(encoded)).cpu().data.numpy()
            else:
                latent_vector = encoded.cpu().data.numpy()
            x = np.r_[x, latent_vector]
            y = np.r_[y, labels.cpu().data.numpy()]
            if i%20 == 0:
                print(f'Progress: [{i}/{len(data_loader)}]')

        if sampling_ratio != 1.:
            idx = np.random.choice(np.arange(len(y)), int(len(y)*sampling_ratio), replace=False)
            x = x[idx]
            y = y[idx]

        data = {'x': x, 'y': y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return x, y

def build_input(args, inputs):
    return inputs.to(args.device) if args.cae else inputs.view(inputs.size(0), -1).to(args.device)

def _gap(inputs):
    return nn.AdaptiveAvgPool2d((1, 1))(inputs)
