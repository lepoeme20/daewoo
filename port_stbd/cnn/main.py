import argparse
import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from config import load_config
from resnet import ResNet18
from trainer import Trainer
from utils import fix_seed


def main(args):
    fix_seed(args.seed)
    model = ResNet18(num_classes=args.num_classes, pretrained=args.pretrain)
    trainer = Trainer(config=vars(args), model=model)

    if args.mode == "train":
        t = time.time()
        trainer.train()
        train_time = time.time() - t
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        print()
        print("Training Finished.")
        print("** Total Time: {}-hour {}-minute".format(int(h), int(m)))

    else:
        test_model = ResNet18(num_classes=args.num_classes, pretrained=args.pretrain)
        state_dict = torch.load(glob.glob(os.path.join(args.ckpt_path, "*.pt"))[0])

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():  # dataparallel processing
            if "module" in k:
                k = k.replace("module.", "")
            new_state_dict[k] = v

        test_model.load_state_dict(new_state_dict)
        trainer.test(test_model)


if __name__ == "__main__":
    args = load_config()
    main(args)
