import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import cv2

from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import itertools  

from utils.preprocess_colorization import calculate_cdf,calculate_lookup,match_histograms


class BuildDataset(Dataset):
    def __init__(self, df,transform,image_size,sampling_ratio,hist_option):
        
        #df = trn
        #sampling_ratio=0.1
        self.image_size = image_size
        self.hist_option = hist_option
        self.img_path = df['image'].values
        self.labels = df['height'].values 
       
        self.sampling_ratio = sampling_ratio
        r_ratio = 1/sampling_ratio
        use_idx = [ i for i in range(len(self.labels)) if i%r_ratio ==0]
        
        img_path_list = df.iloc[use_idx,df.columns=='image'].values.tolist()
        img_path_list = list(itertools.chain(*img_path_list))
        
        label_path_list = df.iloc[use_idx,df.columns=='height'].values.tolist()
        label_path_list = list(itertools.chain(*label_path_list))
        
        self.img_path = img_path_list
        self.labels = label_path_list
        
        self.transform = transform
      
       
     
    def __len__(self):
     
        return len(self.labels)
    
    def __getitem__(self, idx):
       
        img_path = self.img_path[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
       
   
        ff = np.fromfile(img_path, np.uint8)
        frame = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED) # img = array
        if self.hist_option == 'he' :
            
            frame = cv2.equalizeHist(frame) 
            
        elif self.hist_option == 'clahe' :
            #print ("use clahe")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            frame = clahe.apply(frame)
        elif self.hist_option == 'matching_gray':
            frame = cv2.imread(img_path)
            #SOURCE_IMAGE = "E:/crop/20200510120104_34.jpg"
            REFERENCE_IMAGE = "E:/crop/exmple3_gray.jpg"
            
            image_ref = cv2.imread((REFERENCE_IMAGE))
            output_image = match_histograms(frame, image_ref)
            frame = output_image[:,:,0]
            #plt.imshow(output_image[:,:,0],cmap='gray')
            #image_ref2 = cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB)
            #print('original image')
            
        elif self.hist_option == 'matching_color':
            frame = cv2.imread(img_path)
            #SOURCE_IMAGE = "E:/crop/20200510120104_34.jpg"
            REFERENCE_IMAGE = "E:/crop/eample3.jpg"
            
            image_ref = cv2.imread((REFERENCE_IMAGE))
            image_ref2 = cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB)
            output_image = match_histograms(frame, image_ref2)
            frame = output_image
            #plt.imshow(output_image)
            #image_ref2 = cv2.cvtColor(image_ref,cv2.COLOR_BGR2RGB)
            #print('original image')        
  
        elif self.hist_option == 'matching_gs':
            ## to be completed
            print('gs')
        
        if len(frame.shape) == 2 :
            frame = frame[:,:,np.newaxis ]
            frame = np.repeat(frame,3,axis=2)
            
        
      
        output = self.get_transform(frame)
        
        return output, label
   
    def get_transform(self, frame):
        if self.transform == 0:
            #print('transform 0')
            transform = transforms.Compose([
                transforms.ToPILImage(), #위치바꾸기
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),  #0~1사이로 바꾼다는데?
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.transform == 1:
            #print('transform 1')
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.3352], std=[0.0647]),
            ])
        elif self.transform == 2 :
            # z score scaling
            #print('transform 2')
            frame = frame.astype(np.float32)
            mean = frame.mean()
            std = frame.std()

            # normalize 진행
            frame -= mean
            frame /= std
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])
    
        
        return transform(frame)    

        
        