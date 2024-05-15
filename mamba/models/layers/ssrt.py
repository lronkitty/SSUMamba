import math
from numpy import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchtyping import TensorType
from collections import namedtuple
from typeguard import typechecked
from mamba.models.layers.ssrt_modules import BasicLayer
from .combinations import *
from .se_net import ResBlockSqEx
"""
This may change significantly as I work out how to implement this properly, but until large portions of this are copied from Phil Wang (@lucidrains)
"""

def add_list(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

def tensor3d_to_list(a):
    channel = a.shape[1]
    b = torch.transpose(a, 1, 2)
    b = b.contiguous().view(b.shape[0],b.shape[1]*b.shape[2],b.shape[3],b.shape[4])
    b = list(torch.split(b, channel, dim=1))
    return b

def list_to_tensor3d(a):
    b = torch.stack(a, dim=2)
    return b

def as_list(a):
    b = torch.transpose(a, 2, 0)
    b = torch.transpose(b, 1, 2)
    return b

def as_tensor(a):
    b = torch.transpose(a, 1,2)
    b = torch.transpose(b, 0,2)
    # b = a.permute(1,2,0)
    return b

SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
StateTensor = TensorType['batch', 'state_len', 'state_dim']

@typechecked
class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x: SeqTensor, state: StateTensor) -> StateTensor:
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)

class RecurrentWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,gate='cat',input_resolution=(64,64),proj_s=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias

        self.bias_table_xs = nn.Parameter(torch.zeros(1,window_size[0]*window_size[1]))
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_cross_s = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//2))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_self_s = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//2))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_cross_x = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//2))  # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table_self_x = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads//2))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.state_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.embed_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.gate = gate
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_s = proj_s
        if gate == 'cat':
            if self.proj_s is True:
                self.proj_state = nn.Linear(dim, dim)
        elif gate == 'sru':
            self.proj_state = nn.Linear(dim,dim)
            self.proj_reset = nn.Linear(dim//2,dim)
            self.proj_forget = nn.Linear(dim//2,dim)
        elif gate == 'qru':
            self.proj_self_s = nn.Linear(dim//2, dim)
            self.proj_cross_s = nn.Linear(dim,dim)
        elif gate == 'lstm':
            self.gate_main = nn.Linear(dim,dim)
            self.gate_forget = nn.Linear(dim,dim)
            self.gate_input = nn.Linear(dim,dim)

        self.proj_x = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # self.relative_position_bias_table=trunc_normal_(self.relative_position_bias_table, std=.02)

        self.relative_position_bias_table_cross_s=trunc_normal_(self.relative_position_bias_table_cross_s, std=.02)
        self.relative_position_bias_table_self_s=trunc_normal_(self.relative_position_bias_table_self_s, std=.02)
        self.relative_position_bias_table_cross_x=trunc_normal_(self.relative_position_bias_table_cross_x, std=.02)
        self.relative_position_bias_table_self_x=trunc_normal_(self.relative_position_bias_table_self_x, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_x, state_x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = input_x.shape
        # bias_table_xs = self.bias_table_xs.view(B_,self.window_size[0]*self.window_size[1],1)
        bias_table_xs = self.bias_table_xs.repeat(B_//self.bias_table_xs.shape[0],1)
        input_x = input_x + bias_table_xs.unsqueeze(-1)
        state_linear = self.state_linear(state_x).reshape(B_, N, 4, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        embed_linear = self.embed_linear(input_x).reshape(B_, N, 4, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(input_x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ks, vs, qs, qsx= state_linear[0], state_linear[1], state_linear[2], state_linear[3]  # make torchscript happy (cannot use tensor as tuple)
        kx, vx, qx, qxs= embed_linear[0], embed_linear[1], embed_linear[2], embed_linear[3]  # make torchscript happy (cannot use tensor as tuple)

        #s
        qs = qs * self.scale
        qsx = qsx * self.scale
        #x
        qx = qx * self.scale
        qxs = qxs * self.scale

        #s
        self_attn_s = (qs @ ks.transpose(-2, -1))
        cross_attn_s = (qsx @ kx.transpose(-2, -1))
        #x
        self_attn_x = (qx @ kx.transpose(-2, -1))
        cross_attn_x = (qxs @ ks.transpose(-2, -1))


        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias_cross_s = self.relative_position_bias_table_cross_s[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias_self_s = self.relative_position_bias_table_self_s[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias_cross_x = self.relative_position_bias_table_cross_x[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias_self_x = self.relative_position_bias_table_self_x[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias_cross_s = relative_position_bias_cross_s.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias_self_s = relative_position_bias_self_s.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias_cross_x = relative_position_bias_cross_x.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        relative_position_bias_self_x = relative_position_bias_self_x.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        cross_attn_s = cross_attn_s + relative_position_bias_cross_s.unsqueeze(0)

        self_attn_s = self_attn_s + relative_position_bias_self_s.unsqueeze(0)

        cross_attn_x = cross_attn_x + relative_position_bias_cross_x.unsqueeze(0)

        self_attn_x = self_attn_x + relative_position_bias_self_x.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            cross_attn_s = cross_attn_s.view(B_ // nW, nW, self.num_heads//2, N, N) + mask.unsqueeze(1).unsqueeze(0)
            cross_attn_s = cross_attn_s.view(-1, self.num_heads//2, N, N)
            cross_attn_s = self.softmax(cross_attn_s)
            
            self_attn_s = self_attn_s.view(B_ // nW, nW, self.num_heads//2, N, N) + mask.unsqueeze(1).unsqueeze(0)
            self_attn_s = self_attn_s.view(-1, self.num_heads//2, N, N)
            self_attn_s = self.softmax(self_attn_s)

            cross_attn_x = cross_attn_x.view(B_ // nW, nW, self.num_heads//2, N, N) + mask.unsqueeze(1).unsqueeze(0)
            cross_attn_x = cross_attn_x.view(-1, self.num_heads//2, N, N)
            cross_attn_x = self.softmax(cross_attn_x)

            self_attn_x = self_attn_x.view(B_ // nW, nW, self.num_heads//2, N, N) + mask.unsqueeze(1).unsqueeze(0)
            self_attn_x = self_attn_x.view(-1, self.num_heads//2, N, N)
            self_attn_x = self.softmax(self_attn_x)
        else:
            cross_attn_s = self.softmax(cross_attn_s)
            self_attn_s = self.softmax(self_attn_s)
            cross_attn_x = self.softmax(cross_attn_x)
            self_attn_x = self.softmax(self_attn_x)

        cross_attn_s = self.attn_drop(cross_attn_s)
        self_attn_s = self.attn_drop(self_attn_s)
        cross_attn_x = self.attn_drop(cross_attn_x)
        self_attn_x = self.attn_drop(self_attn_x)

        cross_s = (cross_attn_s @ vx).transpose(1, 2).reshape(B_, N, C//2)
        self_s = (self_attn_s @ vs).transpose(1, 2).reshape(B_, N, C//2)
        cross_x = (cross_attn_x @ vs).transpose(1, 2).reshape(B_, N, C//2)
        self_x = (self_attn_x @ vx).transpose(1, 2).reshape(B_, N, C//2)

        # state_x = self.proj_s(torch.cat((cross_s, self_s), dim=2))

        output_x = self.proj_x(torch.cat((cross_x, self_x), dim=2))

        output_x = self.proj_drop(output_x)

        if self.gate == 'cat':
            if self.proj_s is True:
                state_x = self.proj_state(torch.cat((cross_s, self_s), dim=2))
                state_x = self.proj_drop(state_x)
            return output_x, state_x

        if self.gate == 'sru':
            if self.proj_s is True:
                state_x = self.proj_state(torch.cat((cross_s, self_s), dim=2))
                state_x = self.proj_drop(state_x)
                reset   = self.proj_reset(self_s)
                reset   = self.proj_drop(reset)
                forget  = self.proj_forget(self_s)
                forget  = self.proj_drop(forget)
            return output_x, (state_x, reset, forget)
        
        if self.gate == 'qru':
            if self.proj_s is True:
                cross_s = self.proj_cross_s(torch.cat((cross_s,self_s),dim=2))
                cross_s = self.proj_drop(cross_s)
                self_s = self.proj_self_s(self_s)
                self_s = self.proj_drop(self_s)
        
        if self.gate == 'lstm':
            s = torch.cat((cross_s,self_s),dim=2)
            main = self.proj_drop(self.gate_main(s))
            input = self.proj_drop(self.gate_input(s))
            forget = self.proj_drop(self.gate_forget(s))

            return output_x, (main,input,forget)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 8 * self.dim 
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N*4
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)*4
        # x = self.proj(x)
        flops += N * self.dim * self.dim*4
        return flops

@typechecked


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = round(pow(x.shape[1],1/2)),round(pow(x.shape[1],1/2))
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchUnmerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.inflation = nn.Linear(dim, 2* dim, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.inflation(x)

        x = x.view(B, H, W, C*2)
        c = C // 2
        X = torch.zeros(B, 2*H, 2*W, c).type_as(x)
        X[:, 0::2, 0::2, :] = x[..., :c]  # B H/2 W/2 C
        X[:, 1::2, 0::2, :] = x[..., c:2*c]  # B H/2 W/2 C
        X[:, 0::2, 1::2, :] = x[..., 2*c:3*c]  # B H/2 W/2 C
        X[:, 1::2, 1::2, :] = x[..., 3*c:]  # B H/2 W/2 C
        X = X.view(B, -1, c)  # B H/2*W/2 4*C

        X = self.norm(X)

        return X

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class BlockRecurrentSwinIRBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,gate='pass', norm_layer=nn.LayerNorm,if_mlp_s=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm1_state = norm_layer(dim)

        self.attn = RecurrentWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,input_resolution=input_resolution,gate=gate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_state = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.if_mlp_s = if_mlp_s
        # print(if_mlp_s)
        if self.if_mlp_s:
            self.mlp_state = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.softmax = nn.Softmax(dim=-1)
        self.gate = gate  
    
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
         
    def forward(self, x, state, x_size,x_size_next):
        if self.gate == 'sru':
            (state, c) = state
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        shortcut_state = state
        
        x = self.norm1(x)
        try:
            H, W = x_size
            x = x.view(B, H, W, C)
        except:
            H, W = x_size_next
            x = x.view(B, H, W, C)

        state = self.norm1_state(state)
        state = state.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_state = torch.roll(state, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_state = state

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        state_windows = window_partition(shifted_state, self.window_size)  # nW*B, window_size, window_size, C
        state_windows = state_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.gate == 'cat':
            if self.input_resolution == x_size:
                attn_windows, state_windows = self.attn(x_windows, state_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            else:
                attn_windows, state_windows = self.attn(x_windows, state_windows, mask=self.calculate_mask(x_size).to(x.device))

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            state_windows = state_windows.view(-1, self.window_size, self.window_size, C)

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            state_windows = window_reverse(state_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                state = torch.roll(state_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
                state = state_windows
            x = x.view(B, H * W, C)
            state = state.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            state = shortcut_state + self.drop_path(state)
            state = self.drop_path(self.mlp_state(self.norm2_state(state))) + state 

            return x, state

        elif self.gate == 'sru':
            if self.input_resolution == x_size:
                output_x,  (state_x, reset, forget) = self.attn(x_windows, state_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            else:
                output_x,  (state_x, reset, forget) = self.attn(x_windows, state_windows, mask=self.calculate_mask(x_size).to(x.device))

            # merge windows
            attn_windows = output_x.view(-1, self.window_size, self.window_size, C)
            state_x = state_x.view(-1, self.window_size, self.window_size, C)
            reset = reset.view(-1, self.window_size, self.window_size, C)
            forget = forget.view(-1, self.window_size, self.window_size, C)

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            shifted_state = window_reverse(state_x, self.window_size, H, W)  # B H' W' C
            shifted_reset = window_reverse(reset, self.window_size, H, W)  # B H' W' C
            shifted_forget = window_reverse(forget, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                state = torch.roll(shifted_state, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                reset = torch.roll(shifted_reset, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                forget = torch.roll(shifted_forget, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            state = state.view(B, H * W, C)
            reset = reset.view(B, H * W, C)
            forget = forget.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            f = forget.relu().tanh()
            r = reset.relu().tanh()
            s = state.tanh()
            if c is not None:
                c = torch.mul(f,c) + torch.mul((1-f),s)
            else:
                c = torch.mul((1-f),s)
            s = torch.mul(r,c) + torch.mul((1-r),shortcut_state)
            if self.if_mlp_s:
                s = s + self.drop_path(self.mlp_state(self.norm2_state(s)))
            return x, (s,c)

        elif self.gate == 'qru':
            if self.input_resolution == x_size:
                output_x, (self_s, cross_s) = self.attn(x_windows, state_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            else:
                output_x, (self_s, cross_s) = self.attn(x_windows, state_windows, mask=self.calculate_mask(x_size).to(x.device))

            # merge windows
            attn_windows = output_x.view(-1, self.window_size, self.window_size, C)
            self_s_windows = self_s.view(-1, self.window_size, self.window_size, C)
            cross_s_windows = cross_s.view(-1, self.window_size, self.window_size, C)

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            shifted_self_s = window_reverse(self_s_windows, self.window_size, H, W)  # B H' W' C
            shifted_cross_s = window_reverse(cross_s_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                self_s = torch.roll(shifted_self_s, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                cross_s = torch.roll(shifted_cross_s, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
                self_s = shifted_self_s
                cross_s = shifted_cross_s
            x = x.view(B, H * W, C)
            cross_s = cross_s.view(B, H * W, C)
            self_s = self_s.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            z = cross_s
            f = self_s
            state = f * shortcut_state + (1-f) * z

            if self.if_mlp_s:
                state = state + self.drop_path(self.mlp_state(self.norm2_state(state)))

            return x, state

        elif self.gate == 'lstm':
            if self.input_resolution == x_size:
                output_x, (main,input,forget) = self.attn(x_windows, state_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            else:
                output_x, (main,input,forget) = self.attn(x_windows, state_windows, mask=self.calculate_mask(x_size).to(x.device))

            # merge windows
            attn_windows = output_x.view(-1, self.window_size, self.window_size, C)
            main = main.view(-1, self.window_size, self.window_size, C)
            input = input.view(-1, self.window_size, self.window_size, C)
            forget = forget.view(-1, self.window_size, self.window_size, C)

            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            main = window_reverse(main, self.window_size, H, W)  # B H' W' C
            input = window_reverse(input, self.window_size, H, W)  # B H' W' C
            forget = window_reverse(forget, self.window_size, H, W)  # B H' W' C


            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                main = torch.roll(main, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                input = torch.roll(input, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                forget = torch.roll(forget, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            main = main.view(B, H * W, C)
            input = input.view(B, H * W, C)
            forget = forget.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            f = (forget+1).sigmoid()
            i = (input-1).sigmoid()
            z = main.tanh()
            state = torch.mul(shortcut_state, f) + torch.mul(z, i)
            if self.if_mlp_s:
                state = state + self.drop_path(self.mlp_state(self.norm2_state(state)))
            return x, state

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer_bidir_01(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,gate='cat',if_mlp_s=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            BlockRecurrentSwinIRBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,gate=gate,if_mlp_s=if_mlp_s)
            for i in range(depth)])
            
        self.blocks_back = nn.ModuleList([
            BlockRecurrentSwinIRBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,gate=gate,if_mlp_s=if_mlp_s)
            for i in range(depth)])

        # patch merging layer
        self.downsample_flag = downsample
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        # self.state_init = nn.Parameter(torch.zeros(1, input_resolution[0]*input_resolution[1], dim))
        # self.state_init = trunc_normal_(self.state_init, std=.02)
        # self.state_back_init = nn.Parameter(torch.zeros(1, input_resolution[0]*input_resolution[1], dim))
        # self.state_back_init = trunc_normal_(self.state_back_init, std=.02)
        self.gate = gate

    def forward(self, x_list, x_size,x_size_next, state=None):
        x_list_ = x_list.copy()
        x_list_2 = x_list.copy()
        x_all = list_to_tensor3d(x_list)
        x_mean = torch.mean(x_all, dim=-2)
        if state is None:
            state = x_mean #+ self.state_init
            state_back = x_mean # +self.state_back_init



            # for i in range(state.shape[0]):
            #     state[i] = self.state_init
            #     state_back[i] = self.state_back_init

            if self.gate == 'sru':
                state = (state, None)
                state_back = (state_back, None)

        for blk_id in range(len(self.blocks)): 
            for i in range(len(x_list)):
                if self.use_checkpoint:
                    x_list_[i], state = checkpoint.checkpoint(self.blocks[blk_id], x_list_[i], state, x_size,x_size_next)
                else:
                    x_list_[i], state = self.blocks[blk_id](x_list_[i], state, x_size,x_size_next)
            for i in range(len(x_list)):
                if self.use_checkpoint:
                    x_list_2[i], state_back = checkpoint.checkpoint(self.blocks_back[blk_id], x_list_2[i], state_back, x_size,x_size_next)
                else:
                    x_list_2[-i+1], state_back = self.blocks_back[blk_id](x_list_2[-i+1], state_back, x_size,x_size_next)
        x_list_res = add_list(x_list_, x_list_2)
        if self.downsample_flag == PatchUnmerging:
            for i in range(len(x_list_res)):
                x_list_[i] = self.downsample(x_list_res[i], x_size)
        elif self.downsample_flag == PatchMerging:
            for i in range(len(x_list_res)):
                x_list_[i] = self.downsample(x_list_res[i])
        else:
            return x_list_res, x_list_res
        return x_list_, x_list_res

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BRRSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim,dim_next, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224,img_size_next=224 , patch_size=4, resi_connection='1conv',gate='pass',if_mlp_s=True):
        super(BRRSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        BasicLayer = BasicLayer_bidir_01
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,gate=gate,if_mlp_s=if_mlp_s)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim_next, dim_next, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv3d(dim_next, dim_next // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(dim_next // 4, dim_next // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(dim_next // 4, dim_next, 3, 1, 1))
        
        self.downsample_flag = downsample
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_embed_next = PatchEmbed(
            img_size=img_size_next, patch_size=patch_size, in_chans=0, embed_dim=dim_next,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed_next = PatchUnEmbed(
            img_size=img_size_next, patch_size=patch_size, in_chans=0, embed_dim=dim_next,
            norm_layer=None)
        self.img_size = img_size
        self.img_size_next = img_size_next

    def forward(self, x_list, x_size):
        x_size_next = (round(x_size[0]/(self.img_size[0]/self.img_size_next[0])),round(x_size[1]/(self.img_size[1]/self.img_size_next[1])))
        x_list_, x_list_res = self.residual_group(x_list, x_size,x_size_next)
        x_list__ = []
        for i in range(len(x_list)):
            x_list__.append(self.patch_unembed_next(x_list_[i], x_size_next))
        x_list_after = tensor3d_to_list(self.conv(list_to_tensor3d(x_list__)))
        
        for i in range(len(x_list_after)):
            x_list_after[i] = self.patch_embed_next(x_list_after[i])
            
        if self.downsample_flag == PatchUnmerging:
            for i in range(len(x_list)):
                x_list[i] = self.downsample(x_list[i], (x_size_next[0]//2,x_size_next[1]//2))
      
        elif self.downsample_flag == PatchMerging:
            for i in range(len(x_list)):
                x_list[i] = self.downsample(x_list[i])
        x_list = add_list(x_list, x_list_after)

        return x_list, x_list_res, x_size_next

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class ssrt(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',gate='lstm',if_mlp_s=True,**kwargs):
        super(ssrt, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        patches_resolution_list = []
        dim_list = []
        self.img_size_list = []
        half_layers = -((-self.num_layers)//2)
        self.half_layers = half_layers
        for i_layer in range(half_layers):
            patches_resolution_list.append((patches_resolution[0] // (2 ** i_layer),patches_resolution[1] // (2 ** i_layer)))
            dim_list.append(embed_dim * 2 ** i_layer)
            self.img_size_list.append((img_size[0] // (2 ** i_layer),img_size[1] // (2 ** i_layer)))
        for i_layer in range(half_layers, self.num_layers):
            patches_resolution_list.append((patches_resolution[0] // (2 ** ((self.num_layers-i_layer)-1)),patches_resolution[1] // (2 ** ((self.num_layers-i_layer)-1))))
            dim_list.append(embed_dim * 2 ** ((self.num_layers-i_layer)-1))
            self.img_size_list.append((img_size[0] // (2 ** ((self.num_layers-i_layer)-1)),img_size[1] // (2 ** ((self.num_layers-i_layer)-1))))
        patches_resolution_list.append((patches_resolution[0],patches_resolution[1]))
        dim_list.append(embed_dim)
        self.img_size_list.append(img_size)


        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(half_layers):
            layer = BRRSTB(dim_list[i_layer],
                         dim_next = dim_list[i_layer+1],
                         input_resolution=patches_resolution_list[i_layer],
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=PatchMerging if (i_layer < half_layers - 1) else PatchUnmerging,
                         use_checkpoint=use_checkpoint,
                         img_size=self.img_size_list[i_layer],
                         img_size_next = self.img_size_list[i_layer + 1],
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         gate=gate,if_mlp_s=if_mlp_s
                         )
            self.layers.append(layer)
        for i_layer in range(half_layers, self.num_layers):
            layer = BRRSTB(dim_list[i_layer],
                            dim_next = dim_list[i_layer+1],
                            input_resolution=patches_resolution_list[i_layer],
                            depth=depths[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=PatchUnmerging if (i_layer < self.num_layers - 1) else None,
                            use_checkpoint=use_checkpoint,
                            img_size=self.img_size_list[i_layer],
                            img_size_next = self.img_size_list[i_layer + 1],
                            patch_size=patch_size,
                            resi_connection=resi_connection,
                            gate=gate,if_mlp_s=if_mlp_s
                            )          
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        # self.se = ResBlockSqEx(31)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def check_image_size_3d(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0,0,0,0,0,0,0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward_features(self, x_list):
        x_size = (x_list[0].shape[2], x_list[0].shape[3])
        x_list_ = []
        for i in range(len(x_list)):
            x_list_.append(self.patch_embed(x_list[i]))
        # x = self.patch_embed(x)
            if self.ape:
                x_list_[i] = x_list_[i] + self.absolute_pos_embed
            x_list_[i] = self.pos_drop(x_list_[i])
        res_group = []
        for i_layer in range(self.half_layers):
            x_list_, x_list_res,x_size_next = self.layers[i_layer](x_list_, x_size)
            res_group.append(x_list_res)
            x_size = x_size_next

        # import scipy
        # for oo in range(31):
        #     temp = x_list_res[oo]
        #     temp = temp.cpu().numpy()
        #     temp = temp.reshape(128,128,192)
        #     scipy.io.savemat('/home/ironkitty/data/paper3/projects/T3SC/feature_3/'+str(oo)+'.mat', {'data':temp})
            


        for i_layer in range(self.half_layers, self.num_layers):
            x_list_, x_list_res,x_size_next = self.layers[i_layer](add_list(x_list_,res_group[self.num_layers-i_layer-1]), x_size)
            x_size = x_size_next
        x_list__ = []
        for i in range(len(x_list)):
            x_list__.append(self.norm(x_list_[i]))  # B L C
            x_list__[i] = self.patch_unembed(x_list__[i], x_size)
        
        # import scipy
        # for oo in range(31):
        #     temp = x_list__[oo]
        #     temp = temp.cpu().numpy()
        #     temp = temp.reshape(48,512,512)
        #     temp = temp.transpose(1,2,0)
        #     scipy.io.savemat('/home/ironkitty/data/paper3/projects/T3SC/feature_4/'+str(oo)+'.mat', {'data':temp})

        return x_list__

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        H, W = x.shape[2:]
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x = self.check_image_size(x)
            x = x.unsqueeze(1)
            x_first = self.conv_first(x)

            x_list = tensor3d_to_list(x_first)

            # x_first = self.conv_first(x)
            x_list = self.forward_features(x_list)
            x_list = list_to_tensor3d(x_list)

            res = self.conv_after_body(x_list)+x_first
            x = x + self.conv_last(res)
            x = x.squeeze(1)
            
        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
