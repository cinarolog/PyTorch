# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 01:07:23 2022

@author: cinar
"""



#%% import libraries

import torch 
import torch.nn as nn
import fastai
from fastai.vision.all import *
from fastai.metrics import *
from fastai.vision.data import *
from fastai.callback import *
from pathlib import Path
import fastai
fastai.__version__
"""
!pip install fastai --upgrade
"""

#%% import data

path=Path(r"C:/Users/cinar/Desktop/fastai_armed_forces")

data=ImageDataLoaders.from_folder(path,train="train",valid="valid",test="test",
                                  bs=2,item_tfms=Resize(28),shuffle=True)


data.show_batch(figsize=(10,6))




#%% Create Model

fastai_model=nn.Sequential(
    
    nn.Conv2d(3,4,kernel_size=(2,2)),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    
    nn.Conv2d(4,8,kernel_size=(2,2)),
    nn.ReLU(),
    nn.Flatten(),
    
    #lineer katmanlar
    nn.Linear(1152,30),
    nn.ReLU(),
    nn.Linear(30,4)# 4class
    
  
    )

fastai_model


learn=Learner(data,fastai_model,loss_func=fastai.losses.CrossEntropyLossFlat(),
              metrics=[accuracy,error_rate])

learn.summary()

"""
Total params: 34,902
Total trainable params: 34,902
Total non-trainable params: 0

Optimizer used: <function Adam at 0x0000023508895670>
Loss function: FlattenedLoss of CrossEntropyLoss()

Callbacks:
  - TrainEvalCallback
  - Recorder
  - ProgressCallback
"""

#%% Model Train

epoch=20
learn.fit_one_cycle(epoch)



#%% Classic Train

epoch2=5
learn.fit(epoch2,0.001)



#%% Model Train with Fine Tune

epoch3=5

learn.fine_tune(epoch3,freeze_epochs=5)






# print(torch.__version__)
# print(fastai.__version__)





