import argparse

from utils import str2bool


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/hdd/data/daewoo/kmou")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoint")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--mode", type=str, choices=['train', 'test'],default="train")

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=int, default=0.001)
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adam", "adamw"], default="adam"
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--eval_step", type=int, default=20)
    parser.add_argument(
        "--pretrain",
        type=str2bool,
        default="false",
        help="resnet pretrained(true/false)",
    )

    args = parser.parse_args()
    return args
