from urllib.parse import DefragResult
from torch import nn
from typing import Tuple, Union
from models.ec_unetr.dynunet_block import UnetOutBlock, UnetResBlock
#from models.ec_unetr.model_components import UnetrDecoder, StageDecoder, UnetrEncoder,StageDecoder
from monai.networks.layers.utils import get_norm_layer
from models.ec_unetr.dynunet_block import get_conv_layer, UnetResBlock
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from models.ec_unetr.layers import LayerNorm

import numpy as np


class StageEncoder(nn.Module):
    def __init__(self, in_channels,out_channels,input_size,
                 proj_size , depth,  num_heads, spatial_dims=3, 
                 dropout=0.0, transformer_dropout_rate=0.1 ,conv_encoder=True):
        super().__init__()

        #downsample: conv+norm
        self.downsample_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=out_channels),
        )

        #stage blocks
        stage_blocks = []
        if conv_encoder == True:
            stage_blocks.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
                             norm_name='instance', ))
        else:
            #include depth times TransformerBlock   
            '''   
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=input_size, hidden_size=out_channels,
                                                        proj_size=proj_size, num_heads=num_heads,
                                                        dropout_rate=transformer_dropout_rate, pos_embed=True))  
            ''' 
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
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
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

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            '''
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))           
            self.decoder_block.append(nn.Sequential(*stage_blocks))
            '''

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out

class UnetrEncoder(nn.Module):
    def __init__(self, input_size,dims, proj_size, depths,  num_heads=4):
        super().__init__()

        self.encode_blocks=nn.ModuleList()
        self.stage_num=len(input_size)
        for i in range(self.stage_num):
            encoder = StageEncoder(dims[i],dims[i+1],input_size[i],proj_size[i],depths[i],num_heads)
            self.encode_blocks.append(encoder)

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
        hidden_states = []
        for i in range(self.stage_num):
            x=self.encode_blocks[i](x)
            hidden_states.append(x)
        return hidden_states

class UnetrDecoder(nn.Module):
    def __init__(self, input_size,dims, proj_size, depths,  num_heads=4):
        super().__init__()

        input_size=input_size[::-1]
        dims=dims[::-1]

        self.decode_blocks=nn.ModuleList()
        self.stage_num=len(input_size)-1
        for i in range(self.stage_num):
            decoder = StageDecoder(
                spatial_dims=3,
                in_channels=dims[i],
                out_channels=dims[i+1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name='instance',
                out_size=input_size[i+1]
            )
            self.decode_blocks.append(decoder)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, enc_states):
        dec_states=[]
        dec_states.append(enc_states[-1])
        for i in range(self.stage_num):
            dec=self.decode_blocks[i](dec_states[i], enc_states[-i-2])
            dec_states.append(dec)
        return dec_states


class EC_UNETR_Res(nn.Module):

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

        stage=4
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
