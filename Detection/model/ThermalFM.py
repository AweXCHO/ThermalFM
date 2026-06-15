from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as func
from typing import Optional
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
#from mmseg.registry import MODELS
from .my_swin_block import BasicBlock, BasicBlockUp,RadiationModule,TIM, ATM
import torch.nn.functional as F

import importlib
from typing import Tuple, Optional, Union

def _load_weights_to_model(model: torch.nn.Module, weights_path: str, device: torch.device):
    sd = torch.load(weights_path, map_location=device)
    # 常见保存形式处理
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model' in sd:
            sd = sd['model']
    # remove "module." prefix if present
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if k.startswith('module.'):
                nk = k[len('module.'):]
            new_sd[nk] = v
        sd = new_sd
    try:
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        # 尝试直接加载（有些 checkpoint 是整个模型）
        try:
            model = sd
            # if the checkpoint itself is a model object, return it
            return model
        except Exception:
            raise RuntimeError(f"加载权重失败: {e}")
    return model

def compute_params_and_flops(
    model: Optional[torch.nn.Module] = None,
    model_class: Optional[type] = None,
    weights_path: Optional[str] = None,
    input_size: Tuple[int,int,int] = (3, 224, 224),
    device: Optional[Union[str,torch.device]] = None,
    prefer: str = "thop"   # "thop" or "ptflops"
):
    """
    计算参数量与 FLOPs 的主函数。
    - model: 已实例化的 torch.nn.Module（优先使用）
    - model_class: 如果没有提供 model，可以传入模型类（可调用构造），会尝试实例化：model_class()
    - weights_path: 权重文件（.pth/.pt），若提供会加载到 model 上（支持含 'state_dict' 的 checkpoint）
    - input_size: (C,H,W)
    - device: "cuda" 或 "cpu" 或 torch.device
    - prefer: 首选 FLOPs 计算库，"thop" 或 "ptflops"
    返回 dict: {"total_params": int, "trainable_params": int, "flops": float or None, "flops_tool": str or None}
    """
    # 设备选择
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 准备模型实例
    if model is None:
        if model_class is None:
            raise ValueError("需要提供 model 实例或 model_class。")
        model = model_class()
    model.to(device)
    model.eval()

    # 加载权重（如果给了）
    if weights_path is not None:
        model = _load_weights_to_model(model, weights_path, device)
        # 确保在正确设备
        model.to(device)
        model.eval()

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = None
    used_tool = None

    # 先尝试 thop
    if prefer == "thop":
        try:
            thop = importlib.import_module("thop")
            profile = thop.profile
            # thop 有时会修改 model，需要用 deepcopy 或确保无副作用
            input_tensor = torch.randn(1, *input_size).to(device)
            with torch.no_grad():
                macs, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
            # thop 返回的是 MACs，很多文献把 FLOPs = 2 * MACs（multiply + add）
            flops = float(macs) * 2.0
            used_tool = "thop (MACs*2)"
        except Exception:
            # fallback to ptflops
            try:
                ptflops = importlib.import_module("ptflops")
                get_model_complexity_info = ptflops.get_model_complexity_info
                # ptflops 接受 (C,H,W)
                with torch.no_grad():
                    flops_raw, params_raw = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False)
                # ptflops 返回的是 FLOPs（通常以 float 表示）
                flops = float(flops_raw)
                used_tool = "ptflops"
            except Exception:
                flops = None
                used_tool = None

    else:  # prefer == "ptflops"
        try:
            ptflops = importlib.import_module("ptflops")
            get_model_complexity_info = ptflops.get_model_complexity_info
            with torch.no_grad():
                flops_raw, params_raw = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False)
            flops = float(flops_raw)
            used_tool = "ptflops"
        except Exception:
            try:
                thop = importlib.import_module("thop")
                profile = thop.profile
                input_tensor = torch.randn(1, *input_size).to(device)
                with torch.no_grad():
                    macs, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
                flops = float(macs) * 2.0
                used_tool = "thop (MACs*2)"
            except Exception:
                flops = None
                used_tool = None

    result = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "flops": float(flops) if flops is not None else None,
        "flops_tool": used_tool,
        "input_size": tuple(input_size),
        "device": str(device)
    }
    return result

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
class SwinMAE(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 norm_pix_loss=False,depths: tuple = (2, 2, 18, 2), embed_dim: int = 128, num_heads: tuple = (4, 8, 16, 32),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.4, drop_rate: float = 0, attn_drop_rate: float = 0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm: bool = True,whether_load = True, frozen_stages = -1,pretrain_pth='/mnt/dqdisk/swin-base-phy-fre-checkpoint-499.pth'):
        super().__init__()
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


        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)

        self.layers = self.build_layers()


        self.norm_up = norm_layer(embed_dim)
        # self.skip_connection_layers = self.skip_connection()
        self.radiation_module = RadiationModule(dim=in_chans)
        self.TIM_module = TIM(H=img_size, W=img_size)
        self.ATM_module = ATM(channel=in_chans)

        with torch.no_grad():
            tmp_outputs = self.forward(torch.randn(1, 3, 448, 448))
        self.channel = [i.size(1) for i in tmp_outputs]
        del tmp_outputs
        
        self.init_weights()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
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
        
        # # print('radiation_max:', radiation.max())
        # # print('radiation_min:', radiation.min())
        # # print("x_max:", x.max())
        # # print("x_min:", x.min())
        # x = x + radiation
        # tim_output = self.TIM_module(x)
        # atm_output = self.ATM_module(x)
        # fused = torch.cat([atm_output, tim_output], dim=1)
        # x=  self.conv_fusion(fused)
        # x = self.patch_embed(x)
        
        x = self.ATM_module(x)
        x = self.patch_embed(x)
        outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in (0, 1, 2, 3):
                # 保持和原来相同的 shape/order，但作为局部变量存储
                outputs.append(x.permute(0, 3, 1, 2).contiguous())

        return x, outputs
    
    def forward(self, x):
        _, outputs = self.forward_encoder(x)
        
        # concat 3,4
        feat3 = outputs[2]
        feat4 = outputs[3]
        outputs[3] = torch.cat([feat3, feat4], dim=1)
        
        return outputs
    

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
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


    def init_weights(self):
        checkpoint_path = self.pretrain_pth

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = self.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        
        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        print("————————————————")
        print("Load pre-trained checkpoint from: %s" % checkpoint_path)

if __name__ == '__main__':
    # model = swin_mae()
    model = SwinMAE()
    
    res = compute_params_and_flops(model=model, input_size=(3,224,224))
    print(res)
    # imgs = torch.randn(1, 3, 224, 224)
    # print(model)

    # output_feature= model(imgs)
    # for i in range(len(output_feature)):
    #     print(output_feature[i].shape)
