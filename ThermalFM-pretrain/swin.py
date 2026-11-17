"""Checkpoint-compatible Swin layers shared by pretraining and downstream tasks."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_c: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"images must be [B,C,H,W], got {tuple(x.shape)}")
        _, _, height, width = x.shape
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(torch.cat((x0, x1, x2, x3), dim=-1)))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        shift: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dim must be divisible by num_heads")
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.shift_size = window_size // 2 if shift else 0

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        size = self.window_size
        x = x.view(batch, height // size, size, width // size, size, channels)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, channels)

    def _shift_mask(self, height: int, width: int, device) -> torch.Tensor:
        size = self.window_size
        shift = self.shift_size
        image_mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -size), slice(-size, -shift), slice(-shift, None))
        w_slices = (slice(0, -size), slice(-size, -shift), slice(-shift, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = count
                count += 1
        windows = self._window_partition(image_mask).flatten(1, 2).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        size = self.window_size
        pad_h = (-height) % size
        pad_w = (-width) % size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = height + pad_h, width + pad_w

        if self.shift_size:
            x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        windows = self._window_partition(x).flatten(1, 2)
        window_batch, tokens, _ = windows.shape
        head_dim = channels // self.num_heads
        qkv = self.qkv(windows).reshape(
            window_batch, tokens, 3, self.num_heads, head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(tokens, tokens, -1)
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)

        windows_per_image = (padded_h // size) * (padded_w // size)
        if self.shift_size:
            shift_mask = self._shift_mask(padded_h, padded_w, x.device)
            attention = attention.view(
                batch, windows_per_image, self.num_heads, tokens, tokens
            )
            attention = attention + shift_mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.num_heads, tokens, tokens)

        # Mask padded keys. Standard pre-training size 224 needs no padding,
        # so this branch does not alter checkpoint-equivalent inference.
        if pad_h or pad_w:
            valid = torch.zeros((1, padded_h, padded_w, 1), device=x.device)
            valid[:, :height, :width, :] = 1
            if self.shift_size:
                valid = torch.roll(
                    valid,
                    shifts=(-self.shift_size, -self.shift_size),
                    dims=(1, 2),
                )
            valid = self._window_partition(valid).flatten(1, 2).squeeze(-1)
            key_mask = (valid == 0).to(attention.dtype) * -100.0
            attention = attention.view(
                batch, windows_per_image, self.num_heads, tokens, tokens
            )
            attention = attention + key_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            attention = attention.view(-1, self.num_heads, tokens, tokens)

        attention = self.attn_drop(self.softmax(attention))
        windows = (attention @ v).transpose(1, 2).reshape(
            window_batch, size, size, channels
        )
        windows = self.proj_drop(self.proj(windows))
        x = windows.view(
            batch, padded_h // size, padded_w // size, size, size, channels
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, padded_h, padded_w, channels
        )
        if self.shift_size:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        return x[:, :height, :width, :].contiguous()


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            shift=shift,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicBlock(nn.Module):
    def __init__(
        self,
        index: int,
        embed_dim: int = 96,
        window_size: int = 7,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path: float = 0.1,
        norm_layer=nn.LayerNorm,
        patch_merging: bool = True,
    ) -> None:
        super().__init__()
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        start = sum(depths[:index])
        dim = embed_dim * 2**index
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[index],
                    window_size=window_size,
                    shift=bool(block_index % 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=rates[start + block_index],
                    norm_layer=norm_layer,
                )
                for block_index in range(depths[index])
            ]
        )
        self.downsample = (
            PatchMerging(dim=dim, norm_layer=norm_layer)
            if patch_merging
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        b, h, w, _ = x.shape
        x = x.reshape(b, h, w, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, h * 2, w * 2, -1)
        x = self.norm(x)
        return x



class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
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
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x
