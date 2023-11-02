from urllib.parse import DefragResult
from torch import nn
from typing import Tuple, Union
from models.ec_unetr.dynunet_block import UnetOutBlock, UnetResBlock
from models.ec_unetr.model_components import UnetrDecoder, StageDecoder, UnetrEncoder,StageDecoder

import numpy as np

class RG_UNETR(nn.Module):

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
            do_ds=False

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
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        
        if do_ds:
            net_numpool = 3

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            # mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            # weights[~mask] = 0
            weights = weights / weights.sum()
            print(weights)
            self.ds_weights = weights

        data_size=np.array([data_size,int(data_size/2),int(data_size/4),int(data_size/8),int(data_size/16)])
        data_size_3=data_size**3

        proj_size=[64,64,64,64]

        self.do_ds = do_ds
        self.conv_op = conv_op
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        stage=2
        dims=dims[:stage]
        proj_size=proj_size[:stage]
        depths=depths[:stage]
        input_size=data_size_3[1:stage+1]

        dims.insert(0,in_channels)

        self.encoder = UnetrEncoder(dims=dims, proj_size=proj_size, depths=depths, num_heads=num_heads, input_size=input_size)

        self.decoder = UnetrDecoder(dims=dims, proj_size=proj_size, depths=depths, num_heads=num_heads, input_size=input_size)
       
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder2 = StageDecoder(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=data_size_3[0],
            conv_decoder=True,
        )
        
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def forward(self, x_in):
        enc_states=self.encoder(x_in)

        dec_states = self.decoder(enc_states)

        convBlock = self.encoder1(x_in)
        out = self.decoder2(dec_states[-1], convBlock)

        if self.do_ds:
            outs = [self.out1(out), self.out2(dec_states[-1]), self.out3(dec_states[-2])]
            
            #we simply use multi here
            out = self.ds_weights[0] * outs[0]
            for i in range(1, len(outs)):
                if self.ds_weights[i] != 0:
                    out += self.ds_weights[i] * outs[i]
            return out
        else:
            out = self.out1(out)

        return out
