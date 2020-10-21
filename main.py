import torch
import torchvision
from utils.build_dataloader import get_dataloader

csv_path = './brave_data_label.csv'
trn_loader, dev_loader, tst_loader = get_dataloader(
    csv_path,
    batch_size=32,
    iid=True,
    transform=0)

img, label = next(iter(trn_loader))
