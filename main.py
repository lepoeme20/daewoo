import torch
import torchvision
from utils.build_dataloader import get_dataloader

csv_path = './preprocessing/brave_data_label.csv'
trn_loader, dev_loader, tst_loader = get_dataloader(
    csv_path,
    batch_size=32,
    iid=True,
    transform=0)

img, label = next(iter(trn_loader))

# Make mean and std
# mean = 0.
# std = 0.
# nb_samples = 0.
# for idx, (data, _) in enumerate(trn_loader):
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

#     print(idx, len(trn_loader))

# mean /= nb_samples
# std /= nb_samples
# print("mean: " + str(mean))
# print("std: " + str(std))
# print()
