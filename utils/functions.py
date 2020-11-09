import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ae_regressor.model_ae import AE

def create_model(args):
    autoencoder = AE()
    if torch.cuda.is_available():
        autoencoder = nn.DataParallel(autoencoder)

    autoencoder.to(args.device)
    print(f"Model moved to {args.device}")
    return autoencoder


def get_original_data(data, sampling_ratio):
    img_path = data['image'].values
    y = data['label'].values.reshape(-1, 1)
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

    return x, y

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
    return x, y