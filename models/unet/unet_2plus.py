# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet.unet_conv import Unet3_Conv, unetUp_origin
from models.unet.init_weights import init_weights
import numpy as np
from torchvision import models

class UNet_2Plus(nn.Module):
    def __init__(self,in_channels,out_channels,data_size):
        super(UNet_2Plus, self).__init__()
        self.in_channels=in_channels
        self.is_deconv = True
        self.is_batchnorm = True
        self.is_ds = True
        self.feature_scale = 4

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = Unet3_Conv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool3d(kernel_size=2)
        self.conv10 = Unet3_Conv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv20 = Unet3_Conv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv30 = Unet3_Conv(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        self.conv40 = Unet3_Conv(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv3d(filters[0], out_channels, 1)
        self.final_2 = nn.Conv3d(filters[0], out_channels, 1)
        self.final_3 = nn.Conv3d(filters[0], out_channels, 1)
        self.final_4 = nn.Conv3d(filters[0], out_channels, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)#16*(48)
        maxpool0 = self.maxpool0(X_00)#16*(24)
        X_10 = self.conv10(maxpool0)#32*(24)
        maxpool1 = self.maxpool1(X_10)#32*(12)
        X_20 = self.conv20(maxpool1)#64*(12)
        maxpool2 = self.maxpool2(X_20)#64*(6)
        X_30 = self.conv30(maxpool2)#128*(6)
        maxpool3 = self.maxpool3(X_30)#128*(3)
        X_40 = self.conv40(maxpool3)#256*(3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)#16*(48)
        X_11 = self.up_concat11(X_20, X_10)#32*(24)
        X_21 = self.up_concat21(X_30, X_20)#64*(12)
        X_31 = self.up_concat31(X_40, X_30)#128*(6)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)#16*(48)
        X_12 = self.up_concat12(X_21, X_10, X_11)#32*(24)
        X_22 = self.up_concat22(X_31, X_20, X_21)#64*(12)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)#16*(48)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)#32*(24)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)#16*(48)

        # final layer
        final_1 = self.final_1(X_01)#1*(48)
        final_2 = self.final_2(X_02)#1*(48)
        final_3 = self.final_3(X_03)#1*(48)
        final_4 = self.final_4(X_04)#1*(48)

        final = (final_1 + final_2 + final_3 + final_4) / 4#1*(48)
        '''
        #we just need features
        if self.is_ds:
            return F.sigmoid(final)
        else:
            return F.sigmoid(final_4)
        '''
        return final