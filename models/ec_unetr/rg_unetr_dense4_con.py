import torch
from urllib.parse import DefragResult
from torch import nn
from typing import Tuple, Union
from models.ec_unetr.dynunet_block import UnetOutBlock, UnetResBlock

import numpy as np

from base64 import encode
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from models.ec_unetr.layers import LayerNorm
from models.ec_unetr.transformerblock import TransformerBlock
from models.ec_unetr.dynunet_block import get_conv_layer, UnetResBlock

class StageEncoder(nn.Module):
    def __init__(self, in_channels,out_channels,input_size,
                 proj_size =64, depth=3,  num_heads=4, spatial_dims=3, 
                 dropout=0.0, transformer_dropout_rate=0.1 ):
        super().__init__()

        #downsample: conv+norm
        self.downsample_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=out_channels),
        )

        #stage blocks
        stage_blocks = []    
        #include depth times TransformerBlock       
        for j in range(depth):
            stage_blocks.append(TransformerBlock(input_size=input_size, hidden_size=out_channels,
                                                    proj_size=proj_size, num_heads=num_heads,
                                                    dropout_rate=transformer_dropout_rate, pos_embed=True))  
        self.stage_blocks=nn.Sequential(*stage_blocks)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.downsample_layer(x)
        x = self.stage_blocks(x)
        return x

class StageDecoder(nn.Module):
    def     __init__(
            self,            
            in_channels: int,
            out_channels: int,
            num:int=2,
            out_size: int =0,
            spatial_dims: int=3,
            proj_size: int = 64,
            num_heads: int = 4,          
            depth: int = 3,
            kernel_size: Union[Sequence[int], int]=3,
            upsample_kernel_size: Union[Sequence[int], int]=2,
            norm_name: Union[Tuple, str]= "instance",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        middle_channels=out_channels*num
        self.alpha = nn.Parameter(torch.ones(1,middle_channels,1,1,1))
        self.gamma = nn.Parameter(torch.zeros(1,middle_channels,1,1,1))
        self.beta = nn.Parameter(torch.zeros(1,middle_channels,1,1,1))
        self.epsilon = 1e-5

        self.catconv = nn.Sequential(nn.Conv3d(middle_channels, out_channels, kernel_size, stride=1, padding=1),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU(inplace=True), ) 

        if out_size==0:
            self.decoder_block=UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                                norm_name=norm_name, )
        else:
            decoder_block = []
            for j in range(depth):
                decoder_block.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                        proj_size=proj_size, num_heads=num_heads,
                                                        dropout_rate=0.1, pos_embed=True))
            self.decoder_block=nn.Sequential(*decoder_block)        
        self.apply(self._init_weights) 

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, *inp):
        out = self.transp_conv(inp[0])

        for i in range(1,len(inp)):
            out = torch.cat([out, inp[i]], 1)
        
        if False:
            embedding = (out.pow(2).sum((2,3,4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)            
            gate = 1. + torch.tanh(embedding * norm + self.beta)               
            out = gate * out

        out=self.catconv(out)

        out = self.decoder_block(out)
        return out
        
        inp = list(inp) 
        inp[0] = self.transp_conv(inp[0])
        inp = torch.stack(inp)

        if False:
        #if isinstance(self.decoder_block,(UnetResBlock)):
            embedding = (inp.pow(2).sum((2,3,4,5), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)            
            gate = 1. + torch.tanh(embedding * norm + self.beta)               
            inp = gate * inp

        out=inp.sum(0)
        out = self.decoder_block(out)
        return out

class RG_UNETR_Dense4_Con(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            data_size:int,
            feature_size: int = 8,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=[3,3,3,3],
            dims=[16,32,64,128],
            conv_op=nn.Conv3d,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
        """

        super().__init__()
        
        data_size=np.array([data_size,int(data_size/2),int(data_size/4),int(data_size/8),int(data_size/16)])
        input_size=data_size**3

        #self.conv_op = conv_op

        dims.insert(0,feature_size)

        self.conv00 = UnetResBlock(spatial_dims=3,in_channels=in_channels,out_channels=feature_size,
                                   kernel_size=3,stride=1,norm_name=norm_name)
        self.conv10 = StageEncoder(in_channels, dims[1], input_size[1])
        self.conv20 = StageEncoder(dims[1], dims[2], input_size[2])
        self.conv30 = StageEncoder(dims[2], dims[3], input_size[3])
        self.conv40 = StageEncoder(dims[3], dims[4], input_size[4])

        # upsampling
        self.up_concat01 = StageDecoder(dims[1], dims[0])
        self.up_concat11 = StageDecoder(dims[2], dims[1])
        self.up_concat21 = StageDecoder(dims[3], dims[2])
        self.up_concat31 = StageDecoder(dims[4], dims[3],2,input_size[3])

        self.up_concat02 = StageDecoder(dims[1], dims[0],3)
        self.up_concat12 = StageDecoder(dims[2], dims[1],3)
        self.up_concat22 = StageDecoder(dims[3], dims[2],3,input_size[2])

        self.up_concat03 = StageDecoder(dims[1], dims[0],4)
        self.up_concat13 = StageDecoder(dims[2], dims[1],4,input_size[1])

        self.up_concat04 = StageDecoder(dims[1], dims[0],5)

        # final conv (without any concat)
        self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_4 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

        self.gamma = nn.Parameter(torch.ones(4), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in):
        # column : 0
        X_00 = self.conv00(x_in)#8*(48)
        T_10 = self.conv10(x_in)#16*(24)
        T_20 = self.conv20(T_10)#32*(12)
        T_30 = self.conv30(T_20)#64*(6)
        T_40 = self.conv40(T_30)#128*(3)

        # column : 1
        X_01 = self.up_concat01(T_10, X_00)#8*(48)
        X_11 = self.up_concat11(T_20, T_10)#16*(24)
        X_21 = self.up_concat21(T_30, T_20)#32*(12)
        X_31 = self.up_concat31(T_40, T_30)#64*(6)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)#8*(48)
        X_12 = self.up_concat12(X_21, T_10, X_11)#16*(24)
        X_22 = self.up_concat22(X_31, T_20, X_21)#32*(12)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)#8*(48)
        X_13 = self.up_concat13(X_22, T_10, X_11, X_12)#16*(24)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)#8*(48)

        # out layer
        out_1 = self.out_1(X_01)#*self.gamma[0]#1*(48)
        out_2 = self.out_2(X_02)#*self.gamma[1]#1*(48)
        out_3 = self.out_3(X_03)#*self.gamma[2]#1*(48)
        out_4 = self.out_4(X_04)#*self.gamma[3]#1*(48)

        out = (out_1 + out_2 +out_3 + out_4) / 4#1*(48)
        return out
