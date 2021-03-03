import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from glob import glob
import pickle

def get_files(data_path, name):
    print("**load files**")
    files = sorted(glob(f"{data_path}/*/*"))
    folder_num = len(files)
    if name == 'lee':
        return files[:int(folder_num*0.33)]
    elif name == 'choi':
        return files[int(folder_num*0.33): int(folder_num*0.66)]
    else:
        return files[int(folder_num*0.66):]


def get_img_idx(files, interval=10):
    img_name = [p.split('/')[-1].split('_')[0][:-2] for p in files]
    print("**Convert str to datetime**")
    img_time = [datetime.strptime(img.split("_")[0], '%Y%m%d%H%M') for img in img_name]
    time = img_time[0]
    idx = []
    print("**get index**")
    while time < img_time[-1]:
        print(time)
        try:
            idx.append(np.where(np.array(img_time) == time)[0][0])
        except:
            pass
        time += timedelta(minutes=interval)

def show_image(files, idx, folder_name, time):
    '''
    Display a single image
    '''
    folder_name = folder_name[idx]
    time = time[idx]
    img = cv2.imread(files[idx], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Folder: {folder_name}, Time: {time}")
    # plt.show()
    print(f"Folder: {folder_name}, Time: {time}, Idx: {idx}")

def restore_idx_iter(idx_iter, idx, restart_num=0):
    n = np.where(np.array(idx) == restart_num)[0][0]
    for _ in range(n+1):
        idx_iter.__next__()
    return idx_iter

def n_iter(idx_iter, idx, n_iter=10):
    for _ in range(n_iter):
        idx_iter.__next__()
    return idx_iter

# set data_path
data_path = '/media/lepoeme20/Data/projects/daewoo/brave/data'
name = 'seo'
files = get_files(data_path, name)
folder_name = [p.split('/')[-2] for p in files]
time = [p.split('/')[-1] for p in files]

with open(f'./{name}_idx.pkl', 'rb') as f:
    idx = pickle.load(f)

idx_iter = iter(idx)
idx_iter = restore_idx_iter(idx_iter, idx, restart_num=1592852)
idx_iter = n_iter(idx_iter, idx, n_iter=30)
show_image(files, next(iter(idx_iter)), folder_name, time)