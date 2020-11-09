import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ae_regressor.model_ae import AE

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

    return x, y