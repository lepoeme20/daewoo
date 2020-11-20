import torch
import argparse
import multiprocessing

def bound_float(f):
    f = float(f)
    assert 0 < f <= 1., '비율은 0과 1사이여야 합니다'
    return f

def set_parser(parser):
    base_args = parser.add_argument_group('common arguments')
    base_args.add_argument(
        "--csv_path", type=str, default='./preprocessing/brave_data_label.csv',
        help="csv file path"
    )
    base_args.add_argument(
        "--iid", action="store_true", default=False, help="use argument for iid condition"
    )
    base_args.add_argument(
        "--test", action="store_true", default=False, help="Perform Test only"
    )
    base_args.add_argument(
        "--norm-type", type=int, choices=[0, 1, 2],
        help="0: ToTensor, 1: Ordinary image normalizaeion, 2: Image by Image normalization"
    )
    base_args.add_argument(
        "--batch-size", type=int, help="Batch size"
    )
    base_args.add_argument(
        "--seed", type=int, default=22, help="seed number"
    )
    base_args.add_argument(
        "--label-type", type=int, choices=[0, 1, 2], 
        help="0: Height, 1: Direction, 2: Period"
    )
    ae_args = parser.add_argument_group('Auto Encoder arguments')
    ae_args.add_argument(
        "--img-size", type=int, default=32, help='image size for Auto-encoder (default: 32x32)'
    )
    ae_args.add_argument(
        "--epochs", type=int, default=50, help="# of training epochs"
    )
    ae_args.add_argument(
        "--log-interval", type=int, default=200, help="Set interval for logging"
    )
    ae_args.add_argument(
        "--cae", action="store_false", default=True, help="CAE or not (linear AE)"
    )
    rg_args = parser.add_argument_group('Regression arguments')
    rg_args.add_argument(
        "--use-original", action="store_true", default=False, help="using original image vector to regression"
    )
    rg_args.add_argument(
        "--sampling-ratio", type=bound_float, default=0.1, help="Set sampling ratio"
    )
    rg_args.add_argument(
        "--num-parallel", type=int, default=32, help="Set the # of process for regression"
    )
    return parser

def get_config():
    """set arguments
    Returns:
        args -- [description]
    """
    parser = argparse.ArgumentParser(description="AE + SVR")
    args, _ = set_parser(parser).parse_known_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.data_type = 'iid' if args.iid else 'time'
    args.ae_type = 'cae' if args.cae else 'ae'

    assert args.label_type in [0, 1, 2], 'You have to set task using --label-type'
    if args.label_type == 0:
        print(" Set label as height")
        args.label_type == 'height'
    elif args.label_type == 1:
        print(" Set label as direction")
        args.label_type == 'direction'
    elif args.label_type == 2:
        print(" Set label as period")
        args.label_type == 'period'
    return args