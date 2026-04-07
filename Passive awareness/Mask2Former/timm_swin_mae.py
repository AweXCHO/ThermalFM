from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as func
from typing import Optional
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from typing import Sequence, Dict
import _utils as utils
#from mmseg.registry import MODELS

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_size = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_size, coords_size]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        img_mask = torch.zeros((1, H, W, 1), device=x.device)
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

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, H, W, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self.create_mask(x)
        else:
            mask = None

        x = self.window_partition(x)
        Bn, Mh, Mw, _ = x.shape
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
        qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C', Nh=H // Mh, Nw=H // Mw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        return x
    

class ATM(nn.Module):    
    def __init__(self, channel):
        super(ATM, self).__init__()
        self.channel = channel
        self.conv_weight1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_weight2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv_d = nn.Conv2d(channel, channel, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv_weight1.weight)
        nn.init.zeros_(self.conv_weight2.weight)
        
        nn.init.zeros_(self.conv_d.weight)
        nn.init.ones_(self.conv_d.bias)

    def forward(self, x):
        weight1 = self.conv_weight1(x)
        weight2 = self.conv_weight2(x)
        d = self.conv_d(x)

        y = x * torch.exp((weight1 + weight2) * d)
        return y
    
class TIM(nn.Module):
    def __init__(self, H, W, device=None):
        super(TIM, self).__init__()
        # save dimensions
        self.H = H
        self.W = W
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initialize parameters with shape (3, H, W)
        self.weight = nn.Parameter(torch.zeros(3, H, W, device=self.device))
        self.alpha  = nn.Parameter(torch.ones(3, H, W, device=self.device))
        self.Cth    = nn.Parameter(torch.ones(3, H, W, device=self.device))
        self.Gth    = nn.Parameter(torch.ones(3, H, W, device=self.device))
        self.t0     = nn.Parameter(torch.ones(3, H, W, device=self.device))
        self.te     = nn.Parameter(torch.ones(3, H, W, device=self.device))
        self.Phi0   = nn.Parameter(torch.ones(3, H, W, device=self.device))

    def forward(self, input):
        # computing Phi1
        offset = 0.01
        alpha = self.alpha
        Cth   = self.Cth
        Gth   = self.Gth
        t0    = self.t0
        te    = self.te
        Phi0  = self.Phi0

        tau = torch.div(Cth, Gth + offset)
        expt0tau = torch.exp(-torch.div(t0, tau + offset))
        exptetau = torch.exp(-torch.div(te, tau + offset))

        Numerator = (
            input
            + alpha * tau * expt0tau * (exptetau - 1) * Phi0
        )
        Denominator = (
            torch.div(alpha, Gth + offset)
            * (te + tau * exptetau * (torch.exp(te) - 1))
        )
        phi1 = torch.div(Numerator, Denominator + offset)
        phi1 = phi1 * self.weight
        return phi1


class RadiationModule(nn.Module):
    def __init__(self, dim: int):
        super(RadiationModule, self).__init__()
        self.e_conv         = nn.Conv2d(dim, dim, kernel_size=1)
        self.T_conv         = nn.Conv2d(dim, dim, kernel_size=1)
        self.e_norm         = nn.BatchNorm2d(dim)
        self.T_norm         = nn.BatchNorm2d(dim)
        self.e_activation   = nn.Sigmoid()
        self.T_activation   = nn.Sigmoid()
        self.radiation_norm = nn.BatchNorm2d(dim)
        # 可学习的标量放大因子 α（初始化为 1.0）
        self.scale = nn.Parameter(torch.tensor(1.0))
        # 用 1×1 卷积做仿射校准
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.init_radiation_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1）计算 e 和 T
        e = self.e_activation((self.e_conv(x)))
        T = self.T_activation((self.T_conv(x)))
        # 2）Stefan–Boltzmann
        radiation = self.stefan_boltzmann(e, T)
        # 3）BatchNorm
        radiation = self.radiation_norm(radiation)
        # 4）可学习缩放
        radiation = self.scale * radiation
        # 5）1×1 卷积校准
        radiation = self.out_conv(radiation)
        return radiation

    def init_radiation_module(self, init_type='xavier'):
        """
        初始化RadiationModule的权重
        :param module: RadiationModule实例
        :param init_type: 卷积层的初始化方式 ('xavier', 'kaiming', 'normal')
        """
        # 1. 初始化两个主要卷积层 (使用适合Sigmoid的初始化)
        for conv in [self.e_conv, self.T_conv]:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('sigmoid'))
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(conv.weight, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif init_type == 'normal':
                nn.init.normal_(conv.weight, std=0.01)
            else:
                raise ValueError(f"Invalid init_type: {init_type}")
            
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        # 2. 初始化BatchNorm层
        for bn in [self.e_norm, self.T_norm, self.radiation_norm]:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
        
        # 3. 初始化输出卷积层 (使用较小初始化)
        nn.init.normal_(self.out_conv.weight, std=0.01)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)
       
    @staticmethod
    def stefan_boltzmann(e: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        sigma = 5.67
        return sigma * e * T.pow(4)



class BasicBlock(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlock, self).__init__()
        self.index = index
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=(i % 2 == 1),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        self.radiation_module = RadiationModule(dim) if index == 0 else None

        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer) if patch_merging else None

    def forward(self, x):
        for i, layer in enumerate(self.blocks):
                x = layer(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = func.pad(x, (0, self.patch_size - W % self.patch_size,
                             0, self.patch_size - H % self.patch_size,
                             0, 0))
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#@MODELS.register_module()
class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(self):
        self._depth = 5
        self._in_channels = 3
        self._output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = [in_channels] + self._out_channels[1:]

        utils.patch_first_conv(
            model=self, new_in_channels=in_channels, pretrained=pretrained
        )

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        raise NotImplementedError

    def make_dilated(self, output_stride):
        if output_stride not in [8, 16]:
            raise ValueError(f"Output stride should be 16 or 8, got {output_stride}.")

        stages = self.get_stages()
        for stage_stride, stage_modules in stages.items():
            if stage_stride <= output_stride:
                continue

            dilation_rate = stage_stride // output_stride
            for module in stage_modules:
                utils.replace_strides_with_dilation(module, dilation_rate)


class SwinMAE(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, depth = 4, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_channels: int = 49,
                 norm_pix_loss=False, depths: tuple = (2, 2, 18, 2), embed_dim: int = 128, num_heads: tuple = (4, 8, 16, 32),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.4, drop_rate: float = 0, attn_drop_rate: float = 0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm: bool = True,
                 whether_load=True, frozen_stages=-1,
                 pretrain_pth='/mnt/dqdisk/swin-base-phy-fre-checkpoint-499.pth'):
        nn.Module.__init__(self)

        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.weather_load = whether_load
        self.frozen_stages = frozen_stages
        self.pretrain_pth = pretrain_pth

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_channels, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        self.layers = self.build_layers()

        self.norm_up = norm_layer(embed_dim)
        # self.radiation_module = RadiationModule(dim=in_channels)
        # self.TIM_module = TIM(H=img_size, W=img_size)
        self.ATM_module = ATM(channel=49)

        self.init_weights()

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 3)
        return x

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def forward_encoder(self, x):
        # radiation = self.radiation_module(x)
        x = self.ATM_module(x)
        x = self.patch_embed(x)
        self.output = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in (0, 1, 2, 3):
                self.output.append(x.permute(0, 3, 1, 2).contiguous())
        return x

    def forward(self, x):
        # print(x.shape)
        self.forward_encoder(x)
        output = self.output
        
        return tuple(output)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            for _, p in self.named_parameters():
                p.requires_grad = False

    
    def adapt_patch_embed_weights(self, state_dict: dict, target_in_ch: int = 49, key_hint: str = 'patch_embed'):
        """
        把 checkpoint(state_dict) 中的 patch embedding / first conv 权重从原始通道数扩展/截断到 target_in_ch。
        state_dict: 要修改的 dict（会被修改并返回）
        """
        if not isinstance(state_dict, dict):
            raise TypeError("adapt_patch_embed_weights expects a state_dict (dict), got %s" % type(state_dict))

        # 找到可能的 key（兼容不同命名）
        key_name = None
        for k in list(state_dict.keys()):
            if key_hint in k and ('proj.weight' in k or 'patch_embed.proj.weight' in k or 'patch_embed.proj.weight' in k):
                key_name = k
                break
            # 另外常见的命名可能是: 'patch_embed.proj.weight', 'patch_embed.proj.weight' 等
            if 'patch_embed' in k and 'weight' in k and state_dict[k].ndim == 4:
                # 保守判断：4D tensor 且 key 中包含 patch_embed
                key_name = k
                break

        if key_name is None:
            # 未找到 patch_embed 权重，什么也不改
            return state_dict

        w = state_dict[key_name]  # tensor shape (out_c, in_c_pre, k, k)
        if not isinstance(w, torch.Tensor) or w.ndim != 4:
            return state_dict

        pre_in = w.shape[1]
        if pre_in == target_in_ch:
            return state_dict

        # 常用策略：先在 channel 维平均（从 3 -> 1），再 repeat 到 target_in_ch
        if pre_in == 3:
            w_mean = w.mean(dim=1, keepdim=True)   # (out,1,k,k)
            new_w = w_mean.repeat(1, target_in_ch, 1, 1)
            state_dict[key_name] = new_w.clone()
            return state_dict

        # 更通用的扩展/截断策略
        if pre_in < target_in_ch:
            repeats = target_in_ch // pre_in
            rem = target_in_ch % pre_in
            new_w = w.repeat(1, repeats, 1, 1)
            if rem:
                new_w = torch.cat([new_w, w[:, :rem, :, :]], dim=1)
            state_dict[key_name] = new_w[:, :target_in_ch, :, :].clone()
        else:
            # pre_in > target -> 截断
            state_dict[key_name] = w[:, :target_in_ch, :, :].clone()

        return state_dict


    def init_weights(self):
        checkpoint_path = self.pretrain_pth
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # checkpoint 里可能直接是 state_dict 或一个 dict 包含 'model' 键
        checkpoint_model = checkpoint.get('model', checkpoint)

        # 先确保 checkpoint_model 是 dict
        if not isinstance(checkpoint_model, dict):
            raise ValueError("Loaded checkpoint does not contain a state_dict-like dict under 'model' or root.")

        # adapt first conv / patch_embed weights from pretrained (e.g. 3ch) -> target in_chans
        target_in_ch = getattr(self, 'patch_embed', None)
        if target_in_ch is not None:
            target_in_ch = getattr(self.patch_embed, 'in_c', getattr(self, 'in_chans', None))
        if target_in_ch is None:
            # fallback: try to use attribute in_chans or in_ch
            target_in_ch = getattr(self, 'in_chans', getattr(self, 'in_ch', None))
        # set sensible default if still None
        if target_in_ch is None:
            target_in_ch = 3

        checkpoint_model = self.adapt_patch_embed_weights(checkpoint_model, 49)

        # load (allow partial load)
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        print("————————————————")
        print("Load pre-trained checkpoint from: %s" % checkpoint_path)


    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        """Return dict mapping stride -> stage modules"""
        return {
            8: [self.layers[0]],   # feature1: 256, 56x56
            16: [self.layers[1]],   # feature2: 512, 28x28
            32: [self.layers[2]],  # feature3: 1024, 14x14
            32: [self.layers[3]],  # feature4: 1024, 14x14
        }

def swin_mae(**kwargs):
    model = SwinMAE(
        img_size=224, patch_size=4, in_channels=3,
        decoder_embed_dim=1024,
        depths=(2, 2, 18, 2), embed_dim=128, num_heads=(4, 8, 16, 32),
        window_size=7, qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.4, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



if __name__ == '__main__':
    # model = swin_mae()
    model = SwinMAE()
    imgs = torch.randn(1, 3, 224, 224)
    print(model)

    output_feature= model(imgs)
    for i in range(len(output_feature)):
        print(output_feature[i].shape)
