# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:52:02 2022

@author: cinar
"""


#%% import libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import (Dataset,DataLoader)
from skimage import io
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


#%% import data

class veri(Dataset):
    
    def __init__(self,csv_file,root_dir,transforms=None):
        self.annotions=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform= transforms
        
    def __len__(self):
        return len(self.annotions)
    
    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir,self.annotions.iloc[index,0])
        image=io.imread(img_path)
        y_label=torch.tensor(int(self.annotions.iloc[index,1]))
        
        if self.transform:
            image=self.transform(image)  
            
        return (image,y_label)    
        



#%% Data Augmentation

import torchvision.transforms as transforms 

increase_data=transforms.Compose([
    
    transforms.ToPILImage(),
    transforms.Resize((500,500)),
    # transforms.CenterCrop(),
    # transforms.RandomCrop((32,32)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomGrayscale(p=0.2),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    # transforms.GaussianBlur(kernel_size=1),
    # transforms.Grayscale(num_output_channels=3),#RGB
    # transforms.RandomPerspective()
    

    ])


#%% Data Preparing

dataset=veri(
             csv_file=r"C:/Users/cinar/Desktop/armed_forces.csv",
             root_dir=r"C:/Users/cinar/Desktop/armed_forces",
             transforms=increase_data)



#%% Save new images

from torchvision.utils import save_image

count=1

for _ in range(1):
    
    for image,label in dataset:
        save_image(image,"img" + str(count) + "jpg")
        count+=1










