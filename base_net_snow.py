import torch
import torch.nn as nn
from einops import rearrange
import numbers
from timm.models.layers import DropPath,to_2tuple,trunc_normal_
import math
import torch.nn.functional as F
from condconv import *

from distutils.version import LooseVersion
import torchvision
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class Down(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Down,self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
        )
    def forward(self,x):
        x = self.down(x)
        return x

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Up,self).__init__()
        self.up = nn.Sequential(
            nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding))
        )
    def forward(self,x):
        x = self.up(x)
        return x
class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class DWconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DWconv,self).__init__()
        self.dwconv = nn.Conv2d(in_channels,out_channels,3,1,1,bias=False,groups=in_channels)
    def forward(self,x):
        x = self.dwconv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self,chns,factor,dynamic=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if dynamic == False:
            self.channel_map = nn.Sequential(
            nn.Conv2d(chns,chns//factor,1,1,0),
            nn.LeakyReLU(),
            nn.Conv2d(chns//factor,chns,1,1,0),
            nn.Sigmoid()
            )
        else:
            self.channel_map = nn.Sequential(
            CondConv2D(chns,chns//factor,1,1,0),
            nn.LeakyReLU(),
            CondConv2D(chns//factor,chns,1,1,0),
            nn.Sigmoid()
            )
    def forward(self,x):
        avg_pool = self.avg_pool(x)
        map = self.channel_map(avg_pool)
        return x*map


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.act1 = nn.GELU()
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.act2 = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.act1(attn)
        attn = self.conv_spatial(attn)
        attn = self.act2(attn)
        attn = self.conv1(attn)

        return u * attn

class LKA_dynamic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = CondConv2D(dim,dim,5,1,2,1,dim)
        self.act1 = nn.GELU()
        self.conv_spatial = CondConv2D(dim,dim,7,1,9,3,dim)
        self.act2 = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.act1(attn)
        attn = self.conv_spatial(attn)
        attn = self.act2(attn)
        attn = self.conv1(attn)

        return u * attn




class Attention(nn.Module):
    def __init__(self, d_model,dynamic=True):
        super().__init__()

        self.conv11 = nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,groups=d_model)  
        #self.activation = nn.GELU()
        if dynamic == True:
            self.spatial_gating_unit = LKA_dynamic(d_model)
        else:
            self.spatial_gating_unit = LKA(d_model)
        self.conv22 = nn.Conv2d(d_model,d_model,kernel_size=3,stride=1,padding=1,groups=d_model)

    def forward(self, x):
        x = self.conv11(x)
        x = self.spatial_gating_unit(x)
        x = self.conv22(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,VAN=False,dynamic=False):
        super(ConvBlock, self).__init__()
        self.VAN = VAN
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.apply(self._init_weight)
        
        if self.VAN == True:
            if expand_ratio == 1:
                self.conv = nn.Sequential(

                LayerNorm(hidden_dim, 'BiasFree'),
                Attention(hidden_dim,dynamic=dynamic),
            )
            else:
                self.conv = nn.Sequential(
                
                nn.Conv2d(inp, hidden_dim, 1, 1, 0 ),
                LayerNorm(hidden_dim, 'BiasFree'),
                Attention(hidden_dim,dynamic=dynamic),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0),

                )
        else:
            if dynamic == False: 
                if expand_ratio == 1:
                    self.conv = nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        LayerNorm(hidden_dim, 'BiasFree'),
                        nn.GELU(),
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        ChannelAttention(hidden_dim,4,dynamic=dynamic),
                        
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0),

                    )
                else:
                    self.conv = nn.Sequential(
                        # pw
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0 ),

                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        LayerNorm(hidden_dim, 'BiasFree'),
                        nn.GELU(),
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        
                        ChannelAttention(hidden_dim,4,dynamic=dynamic),
                        
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                    )
            else:
                if expand_ratio == 1:
                    self.conv = nn.Sequential(
                        CondConv2D(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        LayerNorm(hidden_dim, 'BiasFree'),
                        nn.GELU(),
                        CondConv2D(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        ChannelAttention(hidden_dim,4,dynamic=dynamic),
                        # pw-linear
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0),

                    )
                else:
                    self.conv = nn.Sequential(
                        
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0 ),

                        CondConv2D(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        LayerNorm(hidden_dim, 'BiasFree'),
                        nn.GELU(),
                        CondConv2D(hidden_dim, hidden_dim, 3, stride, 1,dilation=1, groups=hidden_dim),
                        
                        ChannelAttention(hidden_dim,4,dynamic=dynamic),
                        
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0),
                    )
    def _init_weight(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Conv_block(nn.Module):
    def __init__(self,n,in_channel,out_channele,expand_ratio,VAN=False,dynamic=False):
        super(Conv_block,self).__init__()

        layers=[]
        for i in range(n):
            layers.append(ConvBlock(in_channel,out_channele,1 if i==0 else 1,expand_ratio,VAN=VAN,dynamic=dynamic))
            in_channel = out_channele
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        x = self.model(x)
        return x




