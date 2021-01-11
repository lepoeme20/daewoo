import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import cv2

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ae_regressor.model_ae import AE, CAE

def create_model(args):
    if args.cae:
        autoencoder = CAE()
        print("CAE will be used")
    else:
        autoencoder = AE()
        print("Linear AE will be used")
    if torch.cuda.is_available():
        autoencoder = nn.DataParallel(autoencoder)

    autoencoder.to(args.device)
    print(f"Model moved to {args.device}")
    return autoencoder


def get_original_data(args, data, phase):
    root_path = f'./ae_regressor/data/{args.label_type}/norm_{args.norm_type}/{args.data_type}/img_flatten/original'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'{phase}_seed_{args.seed}.pkl')
    print(data_path)

    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        total_x, y = data['x'], data['y']
    else:
        img_path = data['image'].values
        if args.label_type == 'height':
            y = data['height'].values
        elif args.label_type == 'direction':
            y= data['direction'].values
        else:
            y = data['period'].values
        # x = np.empty([0, 32*32])
        total_x = np.empty([len(img_path), 32*32])

        for i in range(len(img_path)):
            frame = cv2.imread(img_path[i])
            if len(frame.shape) == 3:
                frame = frame[:, :, 0]
            frame = cv2.resize(frame, dsize=(32, 32), interpolation=cv2.INTER_AREA)

            # make a 1-dimensional view of arr
            flat_arr = frame.ravel().reshape(1, -1)
            # x = np.r_[x, flat_arr]
            total_x[i] = flat_arr
            if i%1000 == 0:
                print(f'Progress: [{i}/{len(img_path)}]')

        print("save data")
        data = {'x': total_x, 'y': y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return total_x, y

def get_data(args, data_loader, model, phase):
    print(f"Latent vectors will be extracted on {args.device}")
    root_path = f'./ae_regressor/data/{args.label_type}/norm_{args.norm_type}/{args.data_type}/{args.ae_type}'
    os.makedirs(root_path, exist_ok=True)
    data_path = os.path.join(root_path, f'{phase}_seed_{args.seed}.pkl')

    if os.path.isfile(data_path):
        print("Load data")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        total_x, total_y = data['x'], data['y']
    else:
        print("build data")
        total_x = np.empty([len(data_loader), 64])
        total_y = np.empty([len(data_loader)])

        for i, (inputs, labels) in enumerate((data_loader)):
            encoded = model(build_input(args, inputs))
            if args.cae:
                latent_vector = torch.squeeze(_gap(encoded)).cpu().data.numpy()
            else:
                latent_vector = encoded.cpu().data.numpy()
            total_x[i] = latent_vector
            total_y[i] = labels.cpu().data.numpy()
            if i%20 == 0:
                print(f'Progress: [{i}/{len(data_loader)}]')

        print("save data")
        data = {'x': total_x, 'y': total_y}
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

    return total_x, total_y

def build_input(args, inputs):
    return inputs.to(args.device) if args.cae else inputs.view(inputs.size(0), -1).to(args.device)

def _gap(inputs):
    return nn.AdaptiveAvgPool2d((1, 1))(inputs)

def get_cls_label(labels, dataset):
    q1, q2, q3 = None, None, None
    if dataset == 'brave':
        q1, q2, q3 = 1.17, 1.3, 2.0
    elif dataset == 'weather':
        q1, q2, q3 = 0.591, 0.713, 0.920
    labels[labels < q1] = 0
    labels[labels >= q1] = 1
    labels[labels >= q2] = 2
    labels[labels >= q3] = 3

    return labels.type(torch.LongTensor)
'''
def pred2height(pred, dataset):
    # weather
    # gap: 0.1316, min: 0.304
    if dataset == 'weather':
        height = np.array([0.304, 0.4356, 0.5672, 0.6988, 0.8304, 0.962, 1.0936, 1.2252, 1.3568, 1.4884])
        return torch.tensor(height[pred], dtype=torch.float, device=pred.device)
'''

def pred2height(pred,label_range):
    # weather
    # gap: 0.1316, min: 0.304
    if label_range == 2 :
        step = 0.2
        matching = np.arange(0,2.0,step)
        avg_height = {index : np.round(v+step/2,3)  for index,v in enumerate(matching)}
    
    elif label_range == 1 :
        step = 0.1
        matching = np.arange(0,2.0,step)
    
        avg_height = {index : np.round(v+step/2,3)  for index,v in enumerate(matching)}

    elif label_range == 0 :
        avg_height = np.array([0.304, 0.4356, 0.5672, 0.6988, 0.8304, 0.962, 1.0936, 1.2252, 1.3568, 1.4884])

    else :
        print('error')
    
    result = [avg_height[i.item()] for i in pred]
    
    return result

def plot_confusion_matrix(cm, classes,title,save_path,normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'MAE:{title[0]:.4f}, MAPE:{title[1]:.4f}, ACC:{title[2]:.4f}, SA:{title[3]:.4f}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('{}/cm.png'.format(save_path))
    plt.clf()

def make_pred_plot(real,pred,save_path) :
    plt.plot(pred,label='pred')
    plt.plot(real,label='real')    
    plt.legend()
    plt.savefig(f'{save_path}/plot.jpg')
    plt.clf()
    
def soft_acc(real,pred,window_size) :
    
    cm = confusion_matrix(real,pred)
    correct = cm.diagonal().sum()
    total = cm.sum()
    #acc = correct/total
    
    for i in range(1,window_size) :
        cm_1 = cm[:,i:]
        correct1 = cm_1.diagonal().sum()
        
        cm_2 = cm[i:,:]
        correct2 = cm_2.diagonal().sum()
        correct += (correct1+correct2)
    soft_acc = correct/total
    return soft_acc
