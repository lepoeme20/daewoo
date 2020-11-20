  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.build_dataloader import get_dataloader
import utils.functions as F
from ae_regressor import config

def main(args):
    # Create model
    autoencoder = F.create_model(args)
    # Load data
    trn_loader, dev_loader, tst_loader = get_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        dtype=args.data_type,
        iid=args.iid,
        transform=args.norm_type,
        img_size=args.img_size
    )

    # Set & create save path
    if args.iid:
        print("** Training progress with iid condition **")
        fig_save_path = f'./ae_regressor/save_fig/norm_{args.norm_type}/iid'
        model_save_path = f'./ae_regressor/best_model/{args.ae_type}/norm_{args.norm_type}/iid'
    else:
        print("** Training progress with time series condition **")
        fig_save_path = f'./ae_regressor/save_fig/norm_{args.norm_type}/time'
        model_save_path = f'./ae_regressor/best_model/{args.ae_type}/norm_{args.norm_type}/time'
    os.makedirs(fig_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    if args.test:
        print("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(model_save_path, 'autoencoder.pkl'))
        autoencoder.module.load_state_dict(checkpoint['model'])

        for i, (inputs,_) in enumerate(tst_loader):
            inputs = F.build_input(args, inputs)
            # ============ Forward ============
            criterion = nn.MSELoss()
            _, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            if i % 200 == 0:
                print(f'loss btw test image - reconstructed: {loss:.4f}')
        exit(0)

    else:
        # Define an optimizer and criterion
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.0002)
        best_loss = 1000

        for epoch in range(args.epochs):
            running_loss = 0.0
            _dev_loss = 0.0
            dev_loss = None
            inputs, outputs = None, None

            print('\n\n<Training>')
            for i, (inputs, _) in enumerate(trn_loader):
                inputs = F.build_input(args, inputs)
                # ============ Forward ============
                _, outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                # ============ Backward ============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ============ Logging ============
                running_loss += loss.data
                if i % args.log_interval == args.log_interval-1:
                    print(f'[Trn] {epoch+1}/{args.epochs}, {i+1}/{len(trn_loader)} \
                        loss: {(running_loss/args.log_interval):.5f}')
                    running_loss = 0.0

            # Validate Model
            print('\n\n<Validation>')
            autoencoder.eval()

            for idx, (inputs, _) in enumerate(dev_loader):
                # step progress
                inputs = F.build_input(args, inputs)

                with torch.no_grad():
                    # ============ Forward ============
                    _, outputs = autoencoder(inputs)
                    loss = criterion(outputs, inputs)

                    # ============ Logging ============
                    _dev_loss += loss
                    dev_loss = _dev_loss/(idx+1)
                    if idx % args.log_interval == args.log_interval-1:
                        print(f'[Dev] {epoch+1}/{args.epochs}, {idx+1}/{len(dev_loader)} \
                            loss: {dev_loss:.5f}')

            if dev_loss < best_loss:
                best_loss = dev_loss
                print(f"The best model is saved / Loss: {dev_loss:.5f}")
                torch.save({
                    'model': autoencoder.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trained_epoch': epoch,
                }, os.path.join(model_save_path, 'autoencoder.pkl'))

                save_image(
                    torchvision.utils.make_grid(outputs),
                    os.path.join(fig_save_path, f'reconstructed_epoch_{epoch}.jpg')
                    )
                save_image(
                    torchvision.utils.make_grid(inputs),
                    os.path.join(fig_save_path, f'original_epoch.jpg')
                    )

        print('Finished Training')


if __name__ == '__main__':
    # Set random seed for reproducibility
    h_params = config.get_config()
    SEED = h_params.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    main(h_params)