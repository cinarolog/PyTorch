# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:28:44 2022

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
        


#%% Data Preparing

dataset=veri(csv_file=r"C:/Users/cinar/Desktop/armed_forces.csv",
             root_dir=r"C:/Users/cinar/Desktop/armed_forces",
             transforms=torchvision.transforms.Compose([
                 
                 transforms.ToTensor(),
                 transforms.Resize(size=(28,28)),
                 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),)
                 
                 ]))





#%% Data Preprocessing

train_set,test_set=torch.utils.data.random_split(dataset,[200,19])
train_loader=DataLoader(dataset=train_set,batch_size=1,shuffle=False)
test_loader=DataLoader(dataset=test_set,batch_size=1,shuffle=False)

print(type(train_loader))

#%% Data Visualization

import matplotlib.pyplot as plt
import numpy as np

batch_size=1
classes=["Aircraft","Battleship","Helicopter","Combat Tank"]

# classes=["Ferrari","Mclaren","Mercedes","Redbull"]

def imshow(img):
    img= img/2 + 0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    

dataiter=iter(train_loader)
images,labels= dataiter.next() 

imshow(torchvision.utils.make_grid(images))   

print("".join("%5s" % classes[labels[j]] for j in range (batch_size)))
print(images.size())
 


#%%  Create model architecture

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()

        #Conv2d
        self.conv1=nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(5,5))
        self.conv2=nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3))
        self.conv3=nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2))
        self.conv4=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2))
        
        #MaxPool2d
        self.max=nn.MaxPool2d(kernel_size=(2,2))
        
        #Activation function
        self.func=nn.ELU()
        self.func1=nn.ReLU()
        
        #Linear horizons
        self.fc1=nn.Linear(in_features=32, out_features=50)
        self.fc2=nn.Linear(in_features=50, out_features=50)
        self.fc3=nn.Linear(in_features=50, out_features=100)
        self.fc4=nn.Linear(in_features=100, out_features=4)#4 classımız var
        
    def forward(self,x):
            
        x=self.conv1(x)
        x=self.func(x)
            
        x=self.max(x)
            
        x=self.conv2(x)
        x=self.func(x)
            
        x=self.max(x)
            
        x=self.conv3(x)
        x=self.func(x)
            
        x=self.max(x)
            
        x=self.conv4(x)
        x=self.func(x)
            
            #flatten
        x=x.view(x.size(0),-1)
            
            
            #Neural Network
        x=self.fc1(x)
        x=self.func(x)
        x=self.fc2(x)
        x=self.func(x)
        x=self.fc3(x)
        x=self.func(x)
        x=self.fc4(x)
            
        return x
            
          

    

          
#%% Model Train and Test at GPU

import time
start=time.time()

model_gpu=Net()

device=torch.device("cuda")
model_gpu=model_gpu.to(device)


optimizer=torch.optim.Adam(model_gpu.parameters(),lr=0.001)

error=torch.nn.CrossEntropyLoss()

""" Learning Rate  """
from torch.optim.lr_scheduler import StepLR
lr=StepLR(optimizer, step_size=2, gamma=0.7)


new_epoch=21
count=0
loss_list=[]
acc_list=[]
iteration_list=[]

for i in range(new_epoch):
    
    lr.step()
    
    print("Epoch : ",new_epoch , " LR : ",lr.get_lr())
    
    for i,(images,label) in enumerate(train_loader):
        
        images=images.to(device)
        label=label.to(device)
        
        predict=model_gpu(images)
        optimizer.zero_grad()
        loss=error(predict,label)
        loss.backward()
        optimizer.step()
        
        count+=1
        
        if count % 100 == 0:
            
            total=0
            correct=0
            uncorrect=0
            
            for image,label in test_loader:
                
                images=images.to(device)
                label=label.to(device)
                
                out=model_gpu(images.float())
                pred=torch.max(out.data,1)[1]
                total+=len(label)
                
                correct=(pred==label).sum()
                uncorrect=(pred!=label).sum()
                

            accuracy = 100*(correct/float(total))
            inaccuracy = 100*(uncorrect/float(total))

            loss_list.append(loss.data)
            acc_list.append(accuracy)
            iteration_list.append(count)
            
        if count % 100 == 0:
                
            print("Iteration:{} Loss:{} Accuracy: {}% Error:{}%".format(count,loss.data,accuracy,inaccuracy))
            # print("Iterartion :",count)
            # print("Loss :",loss.data)
            # print("Accuracy :",accuracy)
            # print("Error :",inaccuracy)
 


end=time.time()

print("Süre: ", end - start)


#%% Model Test 

def accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            predict=model(x)
            _,pred=predict.max(1)
            num_correct+=(pred==y).sum()
            num_samples+=pred.size(0)
            
        print(f"Got {(num_correct)} / {(num_samples)} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
            
        model.train()






  
    
    
    
    
    
    
    
    
    
          
            
          
            
          
            
          