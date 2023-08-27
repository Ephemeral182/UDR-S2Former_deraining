import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath,to_2tuple,trunc_normal_
import math 
import time
from base_net_snow import *
import torch.nn.functional as F
if torch.cuda.is_available():
     device = torch.device("cuda:0")
def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)
    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class channel_shuffle(nn.Module):
    def __init__(self,groups=3):
        super(channel_shuffle,self).__init__()
        self.groups = groups
    
    def forward(self,x):
        B,C,H,W = x.shape
        assert C % self.groups == 0
        C_per_group = C // self.groups
        x = x.view(B,self.groups,C_per_group,H,W)
        x = x.transpose(1,2).contiguous()

        x = x.view(B,C,H,W)
        return x

class overlapPatchEmbed(nn.Module):
    def __init__(self,img_size=224,patch_size=7,stride=4,in_channels=3,dim=768):
        super(overlapPatchEmbed,self).__init__()
        
        patch_size=to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels,dim,kernel_size=patch_size,stride=stride,padding=(patch_size[0]//2,patch_size[1]//2))
        self.norm = nn.LayerNorm(dim)
        
        self.apply(self._init_weight)
    
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
    
    def forward(self,x):
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_head}."

        self.dim = dim
        self.num_heads = num_head
        head_dim = dim // num_head
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim,1,1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        B, N, C = x.shape
        x_conv = self.conv(x.reshape(B,H,W,C).permute(0,3,1,2))

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x.transpose(1,2).reshape(B,C,H,W))
        x = self.proj_drop(x)
        x = x+x_conv
        return x


class SimpleGate(nn.Module):
    def forward(self,x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MFFN(nn.Module):
    def __init__(self, dim, FFN_expand=2,norm_layer='WithBias'):
        super(MFFN, self).__init__()

        self.conv1 = nn.Conv2d(dim,dim*FFN_expand,1)
        self.conv33 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,3,1,1,groups=dim*FFN_expand)
        self.conv55 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,5,1,2,groups=dim*FFN_expand)
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(dim,dim,1)

        self.apply(self._init_weights)
    def _init_weights(self,m):
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
        x1 = self.conv1(x)
        x33 = self.conv33(x1)
        x55 = self.conv55(x1)
        x = x1+x33+x55
        x = self.sg(x)
        x = self.conv4(x)
        return x


class SparseSamplingAttention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None, 
            attn_drop=0., proj_drop=0.,
            img_size=(1,1),):
        super().__init__()
        out_dim = dim
        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim or dim
        self.relative_pos_embedding = True
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shift_size = 0
        self.padding_bottom = (self.ws - self.img_size[0] % self.ws) % self.ws
        self.padding_right = (self.ws - self.img_size[1] % self.ws) % self.ws

        self.sampling_biases = nn.Sequential(
            nn.AvgPool2d(kernel_size=window_size, stride=window_size),
            nn.LeakyReLU(), 
            nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
        )
        self.sampling_scales = nn.Sequential(
            nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
            nn.LeakyReLU(), 
            nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
        )

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, out_dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + window_size - 1) * (window_size + window_size - 1), num_heads)) 
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
            coords_flatten = torch.flatten(coords, 1)  
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
            relative_coords[:, :, 0] += self.ws - 1  
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = torch.clip(relative_coords.sum(-1),-1000,1000)  # clip scope
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

        h, w = self.img_size
        h, w = h + self.shift_size + self.padding_bottom, w + self.shift_size + self.padding_right
        image_reference_w = torch.linspace(-1, 1, w)
        image_reference_h = torch.linspace(-1, 1, h)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        window_num_h, window_num_w = window_reference.shape[-2:]
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.ws) * 2 * self.ws / self.ws / (h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.ws) * 2 * self.ws / self.ws / (w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        self.base_coords = (window_reference+coords).to(device)
        self.coords = coords.to(device)

        self.ranking_constraining = True
        self.topk = int(0.8*h*w)
    def get_constraint_matrix(self, x,var_3d):
        b, c, h, w = x.shape
        mask = torch.zeros(b,c,h*w,requires_grad=False).to(x.device)
        index = torch.topk(var_3d,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.).to(x.device)
        constraint_matrix = torch.where(mask>0,torch.full_like(var_3d,1),torch.full_like(var_3d,0.6)) 
        constraint_matrix = constraint_matrix.reshape(b,c,h,w)
        return constraint_matrix
    def get_biases(self, x, num_predict_total, window_num_h, window_num_w):
        b, c, h, w = x.shape
        sampling_biases = self.sampling_biases(x)
        sampling_biases = sampling_biases.reshape(num_predict_total, 2, window_num_h, window_num_w) # 2 denotes the x-axis and y-axis
        sampling_biases[:, 0, ...] = sampling_biases[:, 0, ...] / (w // self.ws)
        sampling_biases[:, 1, ...] = sampling_biases[:, 1, ...] / (h // self.ws)
        return sampling_biases
    def get_scales(self, x, num_predict_total,window_num_h,window_num_w):
        sampling_scales = self.sampling_scales(x)       
        sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
        return sampling_scales
    def transform_coord(self, scales, biases):
        coords_ = self.coords * scales + biases 
        return coords_
    def get_grid_coords(self, coords,num_predict_total,window_num_h,window_num_w):
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.ws*window_num_h, self.ws*window_num_w, 2)
        return sample_coords
    def grid_sample_function(self, x, grid):
        return F.grid_sample(x,grid=grid, padding_mode='zeros', align_corners=True)
    def window_self_attention(self, q, k, v, window_num_h, window_num_w, b):
        #q = window_partition(q.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).reshape(b, self.out_dim, window_num_h*self.ws, window_num_w*self.ws).permute(0,2,3,1),self.ws).reshape(-1,self.ws*self.ws,self.out_dim//self.num_heads,self.num_heads).permute(0,3,1,2)
        #k = window_partition(k.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).reshape(b, self.out_dim, window_num_h*self.ws, window_num_w*self.ws).permute(0,2,3,1),self.ws).reshape(-1,self.ws*self.ws,self.out_dim//self.num_heads,self.num_heads).permute(0,3,1,2)
        #v = window_partition(v.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).reshape(b, self.out_dim, window_num_h*self.ws, window_num_w*self.ws).permute(0,2,3,1),self.ws).reshape(-1,self.ws*self.ws,self.out_dim//self.num_heads,self.num_heads).permute(0,3,1,2)

        q = q.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        k = k.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        v = v.reshape(b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads)
        q_k_dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
            q_k_dots += relative_position_bias.unsqueeze(0)

        attention_map = q_k_dots.softmax(dim=-1)
        out = attention_map @ v

        out = rearrange(out, '(b h_num w_num) h (ws_h ws_w) d -> b (h d) (h_num ws_h) (w_num ws_w)', h=self.num_heads, b=b, h_num=window_num_h, w_num=window_num_w, ws_h=self.ws, ws_w=self.ws)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
    def forward(self, x,var):
        # to same cuda
        self.base_coords = self.base_coords.to(x.device)
        self.coords = self.coords.to(x.device)

        b, c, h, w = x.shape
        x_residual = x
        # padding
        assert h == self.img_size[0]
        assert w == self.img_size[1]
        x = torch.nn.functional.pad(x, (self.shift_size, self.padding_right, self.shift_size, self.padding_bottom))

        window_num_h, window_num_w = self.base_coords.shape[-4], self.base_coords.shape[-2]
        num_predict_total = b * self.num_heads
        coords = self.base_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1)
        # get constraint matrix
        if self.ranking_constraining:
             var_3d = var.reshape(b,c,h*w)#.mean(1).unsqueeze(1)
             constraint_matrix = self.get_constraint_matrix(x,var_3d)

        # aim to regular sampling operator
        x_var = x * constraint_matrix
        # get sampling factors
        sampling_biases = self.get_biases(x_var,num_predict_total,window_num_h,window_num_w)
        sampling_scales = self.get_scales(x_var,num_predict_total,window_num_h,window_num_w)
        # transform coords
        coords = coords + self.transform_coord(sampling_scales[:, :, :, None, :, None], sampling_biases[:, :, :, None, :, None])
        grid_coords = self.get_grid_coords(coords,num_predict_total,window_num_h,window_num_w)#coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.ws*window_num_h, self.ws*window_num_w, 2)
        # get qkv for self-attention
        qkv = self.qkv(x_residual).reshape(b, 3, self.num_heads, self.out_dim // self.num_heads, h, w).transpose(1, 0).reshape(3*b*self.num_heads, self.out_dim // self.num_heads, h, w)
        qkv = torch.nn.functional.pad(qkv, (self.shift_size, self.padding_right, self.shift_size, self.padding_bottom)).reshape(3, b*self.num_heads, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # gridsampling 
        k_sampling = self.grid_sample_function(k.reshape(num_predict_total, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right),grid_coords)
        v_sampling = self.grid_sample_function(v.reshape(num_predict_total, self.out_dim // self.num_heads, h+self.shift_size+self.padding_bottom, w+self.shift_size+self.padding_right),grid_coords)
        # sparse sampling self-attention
        out = self.window_self_attention(q,k_sampling,v_sampling,window_num_h,window_num_w,b)
        
        out = out[:, :, self.shift_size:h+self.shift_size, self.shift_size:w+self.shift_size]
 


        return out


class Local_Reconstruction(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 img_size=(256,256),
                 ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """

        super().__init__()
        self.dim = dim
        window_size = (window_size,window_size)
        self.relative_pos_embedding = True
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            relative_position_index = torch.clip(relative_coords.sum(-1),-1000,1000)  # clip scope
            self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.ranking_modulation = True
        self.topk = int(0.8*self.window_size[0]*self.window_size[1])
    def get_var_modulation(self, x, var):
            B_, N, C = x.shape
            var_self = var @ var.transpose(-2,-1)
            var_self_ = var_self.reshape(B_,N,N)
            mask = torch.zeros(B_,N,N,requires_grad=False).to(x.device)
            index = torch.topk(var_self_,k=self.topk,dim=-1,largest=True)[1]
            mask.scatter_(-1,index,1.).to(x.device)
            modulation_map = torch.where(mask>0,torch.full_like(var_self_,1),torch.full_like(var_self_,1.2)) 
            modulation_map = modulation_map.unsqueeze(1)
            return modulation_map
    def self_attention_modulation(self, x, var_modulation):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        q_k_dots_before = (q @ k.transpose(-2, -1))
        q_k_dots = q_k_dots_before * var_modulation

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            q_k_dots = q_k_dots + relative_position_bias.unsqueeze(0)
        
        attn = self.softmax(q_k_dots)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    def forward(self, x,var):
        if self.ranking_modulation:
              var_modulation = self.get_var_modulation(x,var)

        out = self.self_attention_modulation(x,var_modulation)

        return out


class Reconstruction_Module(nn.Module):

    def __init__(self,
                 latent_dim,
                 dim,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=SparseSamplingAttention,
                 norm_layer=nn.LayerNorm,
                 img_size=(1,1),
                 ):


        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.isattn = attention,
        self.attn = attention(
                              dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              img_size=img_size,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False

        self.gamma1 = 1.0
        self.gamma2 = 1.0
    
    def forward(self, x,var):
            B,H, W,C = x.shape
            shortcut = x
            x = self.norm1(x)
            if self.isattn[0].__name__ == 'Local_Reconstruction':
                x_windows = window_partition(x, self.window_size)
                var = window_partition(var, self.window_size)
                x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
                var = var.view(-1, self.window_size * self.window_size, C)
                attn_windows = self.attn(x_windows,var)
                x = window_reverse(attn_windows, self.window_size, H, W)
            else:
                attn_windows = self.attn(_to_channel_first(x),var)
                x = _to_channel_last(attn_windows)

            x = shortcut + self.drop_path(self.gamma1 * x)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

class Reconstruction_Module_layer(nn.Module):
    def __init__(self,n,latent_dim,in_channel,head,window_size,globalatten=False,img_size=(1,1)):
        super(Reconstruction_Module_layer,self).__init__()

        #layers=[]
        self.globalatten = globalatten
        self.model = nn.ModuleList([
            Reconstruction_Module(
            latent_dim,
            in_channel,
            num_heads=head,
            window_size=window_size,
            attention=Local_Reconstruction if i%2 == 1 and self.globalatten == True else SparseSamplingAttention,
            img_size=img_size,
            )
            for i in range(n)])

    def forward(self,x,var):
        if self.globalatten == True:
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x, var)
        else:
            x = _to_channel_last(x)
            for model in self.model:
                x = model(x,1)
        return x

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel = channel
        self.reduction = reduction
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, channel,reduction=16):
        super(SALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class Refine_Block(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Refine_Block, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

    
class Refine(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(Refine, self).__init__()
        modules_body = []
        modules_body = [Refine_Block(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Refine_stage(nn.Module):
    def __init__(self, n_feat,fusion_dim,  kernel_size, reduction, act, bias, num_cab):
        super(Refine_stage, self).__init__()
        self.refine0 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine1 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine2 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)
        self.refine3 = Refine(fusion_dim, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat[1],fusion_dim,s_factor=2)
        self.up_dec1 = UpSample(n_feat[1],fusion_dim,s_factor=2)

        self.up_enc2 = UpSample(n_feat[2],fusion_dim,s_factor=4)
        self.up_dec2 = UpSample(n_feat[2],fusion_dim,s_factor=4)

        self.up_enc3 = UpSample(n_feat[3],fusion_dim,s_factor=8)
        self.up_dec3 = UpSample(n_feat[3],fusion_dim,s_factor=8)

        layer0=[]
        for i in range(2):
            layer0.append(CALayer(fusion_dim,16))
            layer0.append(SALayer(fusion_dim,16))
        self.conv_enc0 = nn.Sequential(*layer0)

        layer1=[]
        for i in range(2):
            layer1.append(CALayer(fusion_dim,16))
            layer1.append(SALayer(fusion_dim,16))
        self.conv_enc1 = nn.Sequential(*layer1)

        layer2=[]
        for i in range(2):
            layer2.append(CALayer(fusion_dim,16))
            layer2.append(SALayer(fusion_dim,16))
        self.conv_enc2 = nn.Sequential(*layer2)

        layer3=[]
        for i in range(2):
            layer3.append(CALayer(fusion_dim,16))
            layer3.append(SALayer(fusion_dim,16))
        self.conv_enc3 = nn.Sequential(*layer3)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc0(encoder_outs[0] + decoder_outs[3])
        x = self.refine0(x)
       
        x = x + self.conv_enc1(self.up_enc1(encoder_outs[1]) + self.up_dec1(decoder_outs[2]))
        x = self.refine1(x)
        
        x = x + self.conv_enc2(self.up_enc2(encoder_outs[2]) + self.up_dec2(decoder_outs[1]))
        x = self.refine2(x)
        
        x = x + self.conv_enc3(self.up_enc3(encoder_outs[3]) + self.up_dec3(decoder_outs[0]))
        x = self.refine3(x)
        
        return x

class Transformer_block(nn.Module):
    def __init__(self,dim,num_head=8,groups=2,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.,FFN_expand=2,norm_layer='WithBias',act_layer=nn.GELU,l_drop=0.,mlp_ratio=2,drop_path=0.,sr=1):
        super(Transformer_block,self).__init__()
        self.dim=dim*2
        self.num_head = num_head

        self.norm1 = LayerNorm(self.dim//2, norm_layer)
        self.norm3 = LayerNorm(self.dim//2, norm_layer)

        self.attn_nn = Attention(dim=self.dim//2,num_head=num_head,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr)

        self.ffn = MFFN(self.dim//2, FFN_expand=2,norm_layer=nn.LayerNorm)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self,m):
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
        
    def forward(self,x):
        b,c,h,w = x.shape
        b,c,h,w = x.shape
        x =  self.attn_nn(self.norm1(x).reshape(b,c,h*w).transpose(1,2),h,w)
        b,c,h,w = x.shape
        x = self.drop_path(x)
        x = x+self.drop_path(self.ffn(self.norm3(x)))
        return x

class Transformer(nn.Module):
    def __init__(self,
    img_size=(256,256),
    in_channels=3,
    out_cahnnels=3,
    transformer_blocks = 8,
    dim=[16,32,64,128,256],
    window_size = [8,8,8,8],
    patch_size = 64,
    reconstruction_num = [3,6,7,8],
    head = [1,2,4,8,16],
    FFN_expand_=2,
    qkv_bias_=False,
    qk_sacle_=None,
    attn_drop_=0.,
    proj_drop_=0.,
    norm_layer_= 'WithBias',
    act_layer_=nn.GELU,
    l_drop_=0.,
    drop_path_=0.,
    sr=1,
    conv_num = [4,6,7,8],
    expand_ratio = [2,2,2,2],
    VAN = False,
    dynamic_e = False,
    global_atten = True,
    ):
        super(Transformer,self).__init__()
        self.patch_size = patch_size
        
        self.embed = Down(in_channels,dim[0],3,1,1)
        self.encoder_level0 = nn.Sequential(Conv_block(conv_num[0],dim[0],dim[0],expand_ratio=expand_ratio[0],VAN=VAN,dynamic=dynamic_e))

        self.down0 = Down(dim[0],dim[1],3,2,1) ## H//2,W//2
        self.encoder_level1 = nn.Sequential(Conv_block(conv_num[1],dim[1],dim[1],expand_ratio=expand_ratio[1],VAN=VAN,dynamic=dynamic_e))
        
        self.down1 = Down(dim[1],dim[2],3,2,1)  ## H//4,W//4
        self.encoder_level2 = nn.Sequential(Conv_block(conv_num[2],dim[2],dim[2],expand_ratio=expand_ratio[2],VAN=VAN,dynamic=dynamic_e))
        
        self.down2 = Down(dim[2],dim[3],3,2,1)  ## H//8,W//8
        self.encoder_level3 = nn.Sequential(Conv_block(conv_num[3],dim[3],dim[3],expand_ratio=expand_ratio[3],VAN=VAN,dynamic=dynamic_e))
        
        self.down3 = Down(dim[3],dim[4],3,2,1) ## H//16,W//16
        self.latent = nn.Sequential(*[Transformer_block(dim=(dim[4]),num_head=head[4],qkv_bias=qkv_bias_,qk_scale=qk_sacle_,attn_drop=attn_drop_,proj_drop=proj_drop_,FFN_expand=FFN_expand_,norm_layer=norm_layer_,act_layer=act_layer_,l_drop=l_drop_,drop_path=drop_path_,sr=sr) for i in range(transformer_blocks)])
        
        self.up3 = Up((dim[4]),dim[3],4,2,1)
        self.ca3 = CALayer(dim[3]*2,reduction=4)
        self.reduce_chan_level3 = nn.Conv2d(dim[3]*2, dim[3], kernel_size=1, bias=False)
        self.decoder_level3 = Reconstruction_Module_layer(n=reconstruction_num[3],latent_dim=dim[4],in_channel=dim[3],head=head[3],window_size=window_size[3],globalatten=global_atten,img_size=(img_size[0]//8,img_size[1]//8))
        self.up2 = Up(dim[3],dim[2],4,2,1)
        self.ca2 = CALayer(dim[2]*2,reduction=4)
        self.reduce_chan_level2 = nn.Conv2d(dim[2]*2, dim[2], kernel_size=1, bias=False)
        self.decoder_level2 = Reconstruction_Module_layer(n=reconstruction_num[2],latent_dim=dim[4],in_channel=dim[2],head=head[2],window_size=window_size[2],globalatten=global_atten,img_size=(img_size[0]//4,img_size[1]//4))

        self.up1 = Up(dim[2],dim[1],4,2,1)
        self.ca1 = CALayer(dim[1]*2,reduction=4)
        self.reduce_chan_level1 = nn.Conv2d(dim[1]*2, dim[1], kernel_size=1, bias=False)
        self.decoder_level1 = Reconstruction_Module_layer(n=reconstruction_num[1],latent_dim=dim[4],in_channel=dim[1],head=head[1],window_size=window_size[1],globalatten=global_atten,img_size=(img_size[0]//2,img_size[1]//2))

        self.up0 = Up(dim[1],dim[0],4,2,1)
        self.ca0 = CALayer(dim[0]*2,reduction=4)
        self.reduce_chan_level0 = nn.Conv2d(dim[0]*2, dim[0], kernel_size=1, bias=False)
        self.decoder_level0 = Reconstruction_Module_layer(n=reconstruction_num[0],latent_dim=dim[4],in_channel=dim[0],head=head[0],window_size=window_size[0],globalatten=global_atten,img_size=(img_size[0],img_size[1]))
        
        self.refinement = Refine_stage(n_feat=dim,fusion_dim=dim[0],kernel_size=3,reduction=4,act=nn.GELU(),bias=True,num_cab=6)#nn.Sequential(Conv_block(conv_num[0],dim[0],dim[0],expand_ratio=expand_ratio[0],VAN=VAN,dynamic=dynamic_e))

        self.out_final = nn.Conv2d(dim[0],out_cahnnels,kernel_size=3,stride=1,padding=1)
        self.var_conv3 = nn.Sequential(*[nn.Conv2d(dim[3],dim[3],3,1,1), nn.ELU(),nn.Conv2d(dim[3],out_cahnnels,3,1,1), nn.ELU()])
        self.var_conv2 = nn.Sequential(*[nn.Conv2d(dim[2],dim[2],3,1,1), nn.ELU(),nn.Conv2d(dim[2],out_cahnnels,3,1,1), nn.ELU()])
        self.var_conv1 = nn.Sequential(*[nn.Conv2d(dim[1],dim[1],3,1,1), nn.ELU(),nn.Conv2d(dim[1],out_cahnnels,3,1,1), nn.ELU()])
        self.var_conv0 = nn.Sequential(*[nn.Conv2d(dim[0],dim[0],3,1,1), nn.ELU(),nn.Conv2d(dim[0],out_cahnnels,3,1,1), nn.ELU()])
        self.var_conv_final = nn.Sequential(*[nn.Conv2d(dim[0],dim[0],3,1,1), nn.ELU(),nn.Conv2d(dim[0],out_cahnnels,3,1,1), nn.ELU()])

        self.out3 = nn.Conv2d(dim[3],out_cahnnels,kernel_size=3,stride=1,padding=1)
        self.out2 = nn.Conv2d(dim[2],out_cahnnels,kernel_size=3,stride=1,padding=1)
        self.out1 = nn.Conv2d(dim[1],out_cahnnels,kernel_size=3,stride=1,padding=1)
        self.out0 = nn.Conv2d(dim[0],out_cahnnels,kernel_size=3,stride=1,padding=1)
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self,x):
        # x = self.check_image_size(x)
        x1 = nn.functional.interpolate(x, scale_factor=0.5, mode='bicubic')
        x2 = nn.functional.interpolate(x, scale_factor=0.25, mode='bicubic')
        x3 = nn.functional.interpolate(x, scale_factor=0.125, mode='bicubic')
        #x4 = nn.functional.interpolate(y, scale_factor=0.0625, mode='bilinear')
        encoder_item = []
        decoder_item = []
        inp_enc_level0 = self.embed(x)
        inp_enc_level0 = self.encoder_level0(inp_enc_level0)
        encoder_item.append(inp_enc_level0)

        inp_enc_level1 = self.down0(inp_enc_level0)
        inp_enc_level1 = self.encoder_level1(inp_enc_level1)
        encoder_item.append(inp_enc_level1)

        inp_enc_level2 = self.down1(inp_enc_level1)
        inp_enc_level2 = self.encoder_level2(inp_enc_level2)
        encoder_item.append(inp_enc_level2)
        
        inp_enc_level3 = self.down2(inp_enc_level2)
        inp_enc_level3 = self.encoder_level3(inp_enc_level3)
        encoder_item.append(inp_enc_level3)

        out_enc_level4 = self.down3(inp_enc_level3)
        latent = out_enc_level4
        latent = self.latent(out_enc_level4)

        inp_dec_level3 = self.up3(latent)
        inp_dec_level3 = F.upsample(inp_dec_level3,(inp_enc_level3.shape[2],inp_enc_level3.shape[3]),mode="bicubic")
        inp_dec_level3 = torch.cat([inp_dec_level3, inp_enc_level3], 1)
        inp_dec_level3 = self.ca3(inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3,self.var_conv3[0:-2](inp_dec_level3))
        out3 = self.out3(inp_dec_level3)+ x3
        var3 = self.var_conv3(inp_dec_level3)
        out_dec_level3 = _to_channel_first(out_dec_level3)
        decoder_item.append(out_dec_level3)

        inp_dec_level2 = self.up2(out_dec_level3)
        inp_dec_level2 = F.upsample(inp_dec_level2,(inp_enc_level2.shape[2],inp_enc_level2.shape[3]),mode="bicubic")
        inp_dec_level2 = torch.cat([inp_dec_level2, inp_enc_level2], 1)
        inp_dec_level2 = self.ca2(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2,self.var_conv2[0:-2](inp_dec_level2))
        out2 = self.out2(inp_dec_level2) + x2
        var2 = self.var_conv2(inp_dec_level2)
        out_dec_level2 = _to_channel_first(out_dec_level2)
        decoder_item.append(out_dec_level2)

        inp_dec_level1 = self.up1(out_dec_level2)
        inp_dec_level1 = F.upsample(inp_dec_level1,(inp_enc_level1.shape[2],inp_enc_level1.shape[3]),mode="bicubic")
        inp_dec_level1 = torch.cat([inp_dec_level1, inp_enc_level1], 1)
        inp_dec_level1 = self.ca1(inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1,self.var_conv1[0:-2](inp_dec_level1))
        out1 = self.out1(inp_dec_level1) + x1
        var1 = self.var_conv1(inp_dec_level1)
        out_dec_level1 = _to_channel_first(out_dec_level1)
        decoder_item.append(out_dec_level1)

        inp_dec_level0 = self.up0(out_dec_level1)
        inp_dec_level0 = F.upsample(inp_dec_level0,(inp_enc_level0.shape[2],inp_enc_level0.shape[3]),mode="bicubic")
        inp_dec_level0 = torch.cat([inp_dec_level0, inp_enc_level0], 1)
        inp_dec_level0 = self.ca0(inp_dec_level0)
        inp_dec_level0 = self.reduce_chan_level0(inp_dec_level0)
        out_dec_level0 = self.decoder_level0(inp_dec_level0,self.var_conv0[0:-2](inp_dec_level0)) 
        out0 = self.out0(inp_dec_level0) + x
        var0 = self.var_conv0(inp_dec_level0)
        out_dec_level0 = _to_channel_first(out_dec_level0)
        decoder_item.append(out_dec_level0)

        out_dec_level0_refine = self.refinement(out_dec_level0,encoder_item,decoder_item)
        out_final = self.out_final(out_dec_level0_refine) + x
        var = self.var_conv_final(out_dec_level0_refine)
        return [out_final,out0,out1,out2,out3],[var,var0,var1,var2,var3]
# from ptflops import get_model_complexity_info

# model = Transformer(img_size=(256,256)).cuda()
# H,W=256,256
# flops_t, params_t = get_model_complexity_info(model, (3, H,W), as_strings=True, print_per_layer_stat=True)

# print(f"net flops:{flops_t} parameters:{params_t}")
# # model = nn.DataParallel(model)
# x = torch.ones([1,3,H,W]).cuda()
# b = model(x)
# steps=25
# # print(b)
# time_avgs=[]
# memory_avgs=[]
# with torch.no_grad():
#     for step in range(steps):
        
#         torch.cuda.synchronize()
#         start = time.time()
#         result = model(x)
#         torch.cuda.synchronize()
#         time_interval = time.time() - start
#         memory = torch.cuda.max_memory_allocated()
#         if step>5:
#             time_avgs.append(time_interval)
#         #print('run time:',time_interval)
#             memory_avgs.append(memory)
# print('avg time:',np.mean(time_avgs),'fps:',(1/np.mean(time_avgs)),'memory:',(1/np.mean(memory_avgs)),' size:',H,W)
