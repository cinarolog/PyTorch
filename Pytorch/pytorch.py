# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 01:18:17 2022

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
            
            
    
#%% Train Model

import time
start=time.time()

model=Net()

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

error=torch.nn.CrossEntropyLoss()

epoch=20

for i in range(epoch):
    
    for i,(images,label) in enumerate (train_loader):
        
        optimizer.zero_grad()
        
        tahmin= model(images)
        
        loss=error(tahmin,label)
        
        loss.backward() # geri yayılım
        
        optimizer.step()
        
        print("Epoch [{}/{}] , loss:{:.4f}".format(epoch+1,epoch,loss.item()))
        
        

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



#%% Accuracy

print("Train Accuracy :")
accuracy(train_loader,model)

print("Test Accuracy :")
accuracy(test_loader,model)

"""
Train Accuracy :
Got 192 / 200 with accuracy 96.00
Test Accuracy :
Got 13 / 19 with accuracy 68.42
"""


#%% Model Save

torch.save(model,"armed_forces.pth")# .pt  uzantılı olarak da kaydedebilirsin
torch.save(model.state_dict(),"armed_forces2.pth")# model ağırlıları


#%% Model Load

model1=torch.load("armed_forces.pth")

# Accuracy
print("Train Accuracy :")
accuracy(train_loader,model1)

print("Test Accuracy :")
accuracy(test_loader,model1)

"""
Train Accuracy :
Got 192 / 200 with accuracy 96.00

Test Accuracy :
Got 13 / 19 with accuracy 68.42
"""

#%% Model Train and Test

import time
start=time.time()

model2=Net()

optimizer=torch.optim.Adam(model2.parameters(),lr=0.001)

error=torch.nn.CrossEntropyLoss()

new_epoch=21
count=0
loss_list=[]
acc_list=[]
iteration_list=[]

for i in range(new_epoch):
    for i,(images,labels) in enumerate(train_loader):
        
        predict=model2(images)
        optimizer.zero_grad()
        loss=error(predict,label)
        loss.backward()
        optimizer.step()
        
        count+=1
        
        if count % 100 == 0:
            
            total=0
            correct=0
            uncorrect=0
            
            for image,labels in test_loader:
                out=model(images)
                pred=torch.max(out.data,1)[1]
                total+=len(label)
                
                correct=(pred==labels).sum()
                uncorrect=(pred!=labels).sum()
                

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






#%% Summary

from torchsummary import summary

summary(model,inputsize=(3,28,28))
"""

=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            304
├─Conv2d: 1-2                            296
├─Conv2d: 1-3                            528
├─Conv2d: 1-4                            2,080
├─MaxPool2d: 1-5                         --
├─ELU: 1-6                               --
├─ReLU: 1-7                              --
├─Linear: 1-8                            1,650
├─Linear: 1-9                            2,550
├─Linear: 1-10                           5,100
├─Linear: 1-11                           404
=================================================================
Total params: 12,912
Trainable params: 12,912
Non-trainable params: 0
=================================================================
Out[29]: 
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
├─Conv2d: 1-1                            304
├─Conv2d: 1-2                            296
├─Conv2d: 1-3                            528
├─Conv2d: 1-4                            2,080
├─MaxPool2d: 1-5                         --
├─ELU: 1-6                               --
├─ReLU: 1-7                              --
├─Linear: 1-8                            1,650
├─Linear: 1-9                            2,550
├─Linear: 1-10                           5,100
├─Linear: 1-11                           404
=================================================================
Total params: 12,912
Trainable params: 12,912
Non-trainable params: 0
=================================================================

"""

#%% Loss Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.title("Loss Graph")
plt.plot(iteration_list,loss_list,"-o",color="green")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()


#%% Accuracy Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.title("Loss Graph")
plt.plot(iteration_list,acc_list,"-o",color="red")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

#%% Test image

classes=["Aircraft","Battleship","Helicopter","Combat Tank"]

def visualization1(model,image_count=6):
    
    was_training=model.training
    model.eval()
    resim_sayisi=0
    with torch.no_grad():
        for i,(image,labels) in enumerate(test_loader): 
            output=model(image)
            _,predict=torch.max(output,1)
            
            for j in range(image.size()[0]):
                plt.figure(figsize=(40,20))
                resim_sayisi+=1
                ax=plt.subplot(image_count//2,2,resim_sayisi)
                ax.axis("off")
                ax.set_title("Predict : {} ".format(classes[predict[j]]))
                imshow(image.cpu().data[j])
                
                if resim_sayisi== image_count:
                    model.train(mode=was_training)
                    return
    
    
visualization1(model1)


#%% Sequential Model

model3=nn.Sequential(
    
    
    nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(5,5)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2,2)),
    
    nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3)),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=(2,2)),
    
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2,2)),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2)),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=32,out_features=50),
    nn.ReLU(),
    nn.Linear(in_features=50,out_features=50),
    nn.ReLU(),
    nn.Linear(in_features=50,out_features=50),
    nn.ReLU(),
    nn.Linear(in_features=50,out_features=4),

    
    )


model3

#%% model3 Train

import time
start=time.time()

model3=Net()

optimizer=torch.optim.Adam(model3.parameters(),lr=0.001)

error=torch.nn.CrossEntropyLoss()

epoch3=20

for i in range(epoch3):
    
    for i,(images,label) in enumerate (train_loader):
        
        optimizer.zero_grad()
        
        tahmin= model(images)
        
        loss=error(tahmin,label)
        
        loss.backward() # geri yayılım
        
        optimizer.step()
        
        print("Epoch [{}/{}] , loss:{:.4f}".format(epoch+1,epoch,loss.item()))
        
        

end=time.time()

print("Süre: ", end - start)


#%% model3 Test

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



#%% model3 Accuracy

print("Train Accuracy :")
accuracy(train_loader,model3)

print("Test Accuracy :")
accuracy(test_loader,model3)

"""
Train Accuracy :
Got 43 / 200 with accuracy 21.50
Test Accuracy :
Got 6 / 19 with accuracy 31.58
"""


#%% Data Augmentation / Veri arttırma















#%% 














#%%













#%% 















#%%













#%% 















