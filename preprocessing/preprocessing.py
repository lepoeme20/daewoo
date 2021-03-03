import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice

def preprocessing(img_name, data_path, folder, save_path):
    try:
        print(img_name, data_path, folder, save_path)
        img = cv2.imread(os.path.join(data_path, folder, img_name), cv2.IMREAD_GRAYSCALE)
        img_gray = img.astype(np.float32)

        # ROI cropping
        img = img_gray[img_gray.shape[0]-448:, img_gray.shape[1]-448:]
        # Resizing
        img = cv2.resize(img, dsize=(224, 224))

        cv2.imwrite(os.path.join(save_path, folder, img_name), img)
    except AttributeError:
        pass

def get_hist(img):
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
        print(f"{col}: {np.mean(img[:, :, i])}, {np.quantile(img[:, :, i], q=0.25)}, {np.quantile(img[:, :, i], q=0.75)}")

    plt.show()

def show_img(next_img):
    img = cv2.imread(next_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(next_img)
    plt.imshow(img)
    return img

data_path = '/media/lepoeme20/Data/projects/daewoo/kmou/PORT'
check_data = []
for folder in sorted(os.listdir(data_path)):
    if '2020-' in folder:
        try:
            data = sorted(os.listdir(os.path.join(data_path, folder)))[-1]
            check_data.append(os.path.join(data_path, folder, data))
        except IndexError as e:
            print(folder, e)

check_iter = iter(check_data)


img = show_img()
get_hist(img)


###### Label
label_path = '/media/lepoeme20/Data/projects/daewoo/kmou/PORT/image2c'
label_files = os.listdir(label_path)
with open('total_PORT.txt','w') as f:
        f.write('')

header_list = ["time", "height", "period", "direction"]
data_PORT = pd.read_csv('total_PORT.txt', names=header_list)

port_height = data_PORT['height'].values
stbd_height = data_STBD['height'].values


def viz_original(array, label_type):
    print(label_type)
    non_zero_data = np.where(array==0, np.nan, array)

    # bar chart
    plt.figure(figsize=(20, 10))
    plt.scatter(range(len(non_zero_data)), non_zero_data, marker='.', s=3)
    plt.xlabel('Time', fontsize=24)
    plt.ylabel("Height", fontsize=24)
    plt.ylim(-13, 3)
    plt.xticks(fontsize =22)
    plt.yticks(fontsize =22)
    plt.title(label_type, fontdict={'fontsize': 32})
    plt.savefig(label_type + '.png')
    # plt.show()


def min_max(array):
    print(f"min: {np.min(array)}, max: {np.max(array)}, mean: {np.mean(array)}")
    print(np.quantile(array, 0.75))

data_STBD.set_index(data_STBD['time'], inplace=True)
height = data_STBD['height'].values
viz_original(height[:int(len(height)*0.5)], 'STBD')

data_PORT.set_index(data_PORT['time'], inplace=True)
height = data_PORT['height'].values
viz_original(height[:int(len(height)*0.5)], 'PORT')

for data in data_list:
    min_max(data)

label_path = '/media/lepoeme20/Data/projects/daewoo/weather/data/image2c'
label_files = os.listdir(label_path)
with open(os.path.join(label_path, label_files[0]),'rb') as f:
        c = f.read()

from datetime import datetime, timedelta
from tqdm import tqdm
# ship type
ship_type = 'weather' # kmou
# The day ships started moving
movement = {
    # 'weather': {
    #     'start': '2020-11-08',
    #     'end': '2020-11-11',
    # },
    'weather': {
        'start': '2020-11-13',
        'end': '2020-11-18',
    },
    # 'kmou': {
    #     'start': '2020-11-17-1',
    #     'end': '2020-11-19-0'
    # },
    'kmou': {
        'start': '2020-11-23-0',
        'end': '2020-11-24-0'
    }
}

# set path
data_path = f'/media/lepoeme20/Data/projects/daewoo/{ship_type}/data'
# data_path = f'/media/lepoeme20/Data/projects/daewoo/{ship_type}/PORT'
save_path = f'/media/lepoeme20/Data/projects/daewoo/{ship_type}/daytime_data'
data_folders = sorted(os.listdir(data_path))
total_data = []
progress = tqdm(total=len(data_folders), desc="Progress", position=0, leave=False)
for folder in data_folders:
    if folder and movement[ship_type]['start'] <= folder and movement[ship_type]['end'] >= folder:
        files = sorted(os.listdir(os.path.join(data_path, folder)))
        for img in files:
            time = img.split('_')
            if time[0][8:10] > '06' and time[0][8:10] < '18':
                total_data.append(os.path.join(data_path, folder, img))
    progress.update(1)

len(total_data)
iter_data = iter(total_data)
# 2020-11-12-0/20201112130257 weired
img = show_img(next(islice(iter_data, 5000, None)))