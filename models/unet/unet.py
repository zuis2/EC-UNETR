#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import time
from torch.utils import data

from models.unet.init_weights import init_weights

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding_size=1, init_stride=1):
        super(UnetConv3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                    nn.BatchNorm3d(out_size),
                                    nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                    nn.BatchNorm3d(out_size),
                                    nn.ReLU(inplace=True),)


        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp3, self).__init__()
        self.conv = UnetConv3(in_size, out_size, init_stride=1)
        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=3, stride=2, padding=1,output_padding=1)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv(torch.cat([inputs1, outputs2], 1))

### UNet ###
class UNet(nn.Module):

    def __init__(self,in_channels,out_channels,data_size):
        super(UNet,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        act_fn1 = nn.LeakyReLU(0.2, inplace=True)
        act_fn2 = nn.ReLU()

        self.feature_scale=4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(in_channels, filters[0])
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4])

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3])
        self.up_concat3 = UnetUp3(filters[3], filters[2])
        self.up_concat2 = UnetUp3(filters[2], filters[1])
        self.up_concat1 = UnetUp3(filters[1], filters[0])

        self.out = nn.Conv3d(filters[0],self.out_channels, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self,x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        out = self.out(up1)
        return out
    
###

