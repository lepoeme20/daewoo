import argparse
import random

import numpy as np
import torch


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fix_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# define "soft" cross-entropy for wave direction classification
def make_target_dist(target, window_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dist = torch.FloatTensor(target.shape[0], 120)
    target_dist = target_dist.zero_()

    target_dist = target_dist.scatter_(1, target.view(-1, 1), 1)
    for i in range(1, window_size + 1):
        for j in [1, -1]:
            window_idx = target + i * j
            window_idx[window_idx < 0] += 120
            window_idx[window_idx > 119] -= 120
            target_dist = target_dist.scatter_(1, window_idx.view(-1, 1), 1/(i + 1))
    return target_dist.to(device)


def softXEnt(input, target, window_size):
    target_dist = make_target_dist(target, window_size)
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return  -(target_dist * logprobs).sum() / input.shape[0]


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    ref: https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="best_model.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss