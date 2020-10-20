from utils.build_dataloader import get_dataloader
from torchvision.utils import save_image
from model_ae import Autoencoder

# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
import matplotlib.pyplot as plt

# OS
import os
import argparse

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Perform Test only.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()

    # Load data

    csv_path = './preprocessing/brave_data_label.csv'
    trn_loader, dev_loader, tst_loader = get_dataloader(csv_path, iid=False)

    img, label = next(iter(trn_loader))

    if args.test:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./autoencoder_daewoo_time.pkl"))
        dataiter = iter(tst_loader)
        images, labels = dataiter.next()

        total_loss = 0.0
        for i, (inputs,_) in enumerate(tst_loader, 0):
            inputs = get_torch_vars(inputs)
            #print('start testing : {} th'.format(i))

            # ============ Forward ============
            criterion = nn.MSELoss()
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

            if i % 200 == 0:
                print('loss btw test image - reconstructed: {%.4f}' % loss)

        exit(0)

    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    # originial epoch was 100
    for epoch in range(10):
        running_loss = 0.0
        torch.set_grad_enabled(True)
        for i, (inputs,_) in enumerate((trn_loader)):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Validate Model
        print('\n\n<Validation>')
        autoencoder.eval()
        running_loss = 0.0
        torch.set_grad_enabled(False)

        for idx, (inputs, _) in enumerate((dev_loader)):
            # step progress #
            inputs = get_torch_vars(inputs)

            # ============ Forward ============

            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            # ============ Backward ============

            # ============ Logging ============
            running_loss += loss.data
            print(idx, len(dev_loader), 'Loss: %.5f, '
                         % ((running_loss / (idx + 1))))

        print(['Epoch: %d  Loss: %.5f,' \
                             % (epoch, running_loss / (idx + 1))])

        dataiter = iter(dev_loader)
        images, labels = dataiter.next()

        save_image(torchvision.utils.make_grid(images), "./save_fig/original/(time)original_epoch_{}.jpg".format(epoch))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]

        save_image(torchvision.utils.make_grid(decoded_imgs), "./save_fig/decoded/(time)decoded_epoch_{}.jpg".format(epoch))

        #imshow(torchvision.utils.make_grid(decoded_imgs.data))


    print('Finished Training')
    print('Saving Model...')
    torch.save(autoencoder.state_dict(), "./autoencoder_daewoo_time.pkl")


if __name__ == '__main__':
    main()