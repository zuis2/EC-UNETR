import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet.init_weights import *


class Unet3_Conv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(Unet3_Conv, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        if is_deconv:
            self.conv = Unet3_Conv(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = Unet3_Conv(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.nn.Upsample(scale_factor=2, mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Unet3_Conv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)