#!/usr/bin/env python
# coding: utf-8

# Author: J.Lee, KAIST (Korea), 2020.
# 
# Y.Yang, Multi-Dimensional Atomic Imaging Lab, KAIST
# 
# DL augmentation code
# 
# If you use the DL augmentation or related materials, we would appreciate citation to the following paper:
# J. Lee, C. Jeong and Y. Yang, “Single-atom level determination of 3-dimensional surface atomic structure via neural network-assisted atomic electron tomography”, Nature Communications (2021)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import time
from torch.utils import data

import DL_aug as DLa


### checking that cuda is available or not ###
USE_CUDA=torch.cuda.is_available()
DEVICE=torch.device("cuda" if USE_CUDA else "cpu")
print("CUDA: {}".format(USE_CUDA))
###

###########################################################
### input parameter ###
#zuis
INPUT_PATH="/media/vr/4C2CB2AC2CB29106/workspace/dataset/DL/Pt_input"
INPUT_INSIDE_NAME="GF_Vol"

#INPUT_PATH='./Pt_inputdata'
#INPUT_FILE_NAME='Pt_input_1_real_intepolation_zero_padding'
#INPUT_INSIDE_NAME='ESTvol'
data_size=144;

OUTPUT_PATH='./Pt_inputdata'
#OUTPUT_FILE_NAME='./Pt_output_1'
OUTPUT_INSIDE_NAME='output'
###
###########################################################

### generate DL-augmenation model(Unet) ###
aut = DLa.UnetGenerator_3d(in_channels=1,out_channels=1,num_filter=12).to(DEVICE)
print("model contructing: OK!")
###



### loading a previous saved model parameter ###
#PATH = './DL_aug_save_file/DL_aug_test' # FCC Bf5
PATH = './DL_aug_save_file/DL_aug_Pt_FCC_Bf5' # FCC Bf5
#PATH = './DL_aug_save_file/DL_aug_Pt_FCC_Bf3.2' # FCC Bf3.2
#PATH = './DL_aug_save_file/DL_aug_Pt_amorphous_Bf5' # amorphous Bf5
aut.load_state_dict(torch.load(PATH, map_location={'cuda:0': 'cpu'}))
#aut.load_state_dict(torch.load(PATH))
aut.to(DEVICE)

print("loading save file: OK!")
###

criterion = nn.MSELoss()
###
loss_sum_test=0
j=0
ids=[1105,1119,1123,1134,1140,1156,1162,1177,1183,1197]
for id in ids:
    j=j+1
    INPUT_FILE_NAME= "Pt_input_{}".format(id)
    OUTPUT_FILE_NAME= "Pt_output_{}".format(id)
    input_data = scipy.io.loadmat('{}/{}.mat'.format(INPUT_PATH,INPUT_FILE_NAME))[INPUT_INSIDE_NAME];
    aut.eval()
    with torch.no_grad():
        inputs = torch.tensor(input_data).view(-1,1,data_size,data_size,data_size).float().to(DEVICE);
        
        ### choose intensity scale factor
        #inputs = inputs*13  # for Bf5  (FCC+Bf5, amorphous+Bf5)
        #inputs = inputs*8  # for Bf3.2 (FCC+Bf3.2)
        ###
        
        outputs = aut(inputs)

        loss_test = criterion(inputs, outputs)
        print(loss_test)
        loss_sum_test += (loss_test.item())**0.5 # MSE -> RMSE
        print('loss: %.10f ', loss_sum_test/j)

        outputs = outputs.data[0][0].cpu().numpy()
        scipy.io.savemat('{}/{}.mat'.format(OUTPUT_PATH,OUTPUT_FILE_NAME), {'{}'.format(OUTPUT_INSIDE_NAME):outputs}) # save
        

###




