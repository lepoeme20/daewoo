import os
import argparse
import pandas as pd

def change_csv_path(args):
    print("Modifying img path in csv file...")
    label_df = pd.read_csv(args.csv_path)
    col_img_pth = label_df["image"]
    folder_idx = col_img_pth[0].find("use_data") # 'data_crop' is shared folder name
    pth_before = col_img_pth[0][:folder_idx]
    assert args.root_img_path[-1] == "/", "이미지 경로 마지막을 / 으로 끝나게 해 주세요~!"
    label_df["image"] = label_df["image"].str.replace(pth_before, args.root_img_path)
    label_df.to_csv(args.csv_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weather", help="csv file path")
    parser.add_argument(
        "--root-csv-path",
        type=str,
        help="root directory of label csv file",
        default="./preprocessing/",
    )
    parser.add_argument(
        "--root-img-path",
        type=str,
        help="root directory of image folder",
    )

    args = parser.parse_args()

    args.csv_path = os.path.join(args.root_csv_path, f"{args.dataset}_data_label.csv")
    change_csv_path(args)