"""Downstream ThermalFM backbone extracted from AAA_phys pre-training."""

from __future__ import annotations

from functools import partial
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .checkpoint import (
    CheckpointLoadReport,
    CheckpointSource,
    PHYSICS_PREFIXES,
    load_matching_weights,
)
from .physics_modules import (
    AtmosphereInspiredResidualBlock,
    DetectorTypeRouter,
    RadiationInspiredResidualBlock,
    ThermalInertiaInspiredResidualBlock,
)
from .swin_encoder import BasicBlock, PatchEmbedding


class ThermalFMPhysicsBackbone(nn.Module):
    """Swin encoder with the trained detector-aware physics front end.

    The MAE mask, decoder, and reconstruction head are deliberately absent.
    Input is a normalized three-channel image tensor ``[B, 3, H, W]``.

    ``output_mode='post_stage'`` reproduces the historical ThermalFM feature
    interface: stages 0--2 are returned after PatchMerging. Use
    ``'pre_downsample'`` for a conventional stride-4/8/16/32 pyramid.
    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (2, 2, 18, 2),
        embed_dim: int = 128,
        num_heads: Tuple[int, ...] = (4, 8, 16, 32),
        window_size: int = 7,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        channels_first: bool = True,
        output_mode: str = "post_stage",
        detector_router_hidden_dim: int = 0,
        radiation_layer_scale_init: float = 1.0e-4,
        atmosphere_layer_scale_init: float = 1.0e-4,
        atmosphere_optical_depth_bias_init: float = -2.0,
        tim_layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if not out_indices:
            raise ValueError("out_indices cannot be empty")
        if any(index < 0 or index >= len(depths) for index in out_indices):
            raise ValueError("out_indices contains an invalid stage index")
        if tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError("out_indices must be sorted and contain no duplicates")
        if output_mode not in ("post_stage", "pre_downsample"):
            raise ValueError("output_mode must be 'post_stage' or 'pre_downsample'")

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.depths = tuple(depths)
        self.embed_dim = embed_dim
        self.num_heads = tuple(num_heads)
        self.out_indices = tuple(out_indices)
        self.channels_first = channels_first
        self.output_mode = output_mode
        levels = (
            tuple(min(index + 1, len(depths) - 1) for index in out_indices)
            if output_mode == "post_stage"
            else tuple(out_indices)
        )
        self.out_channels = tuple(embed_dim * 2**level for level in levels)
        self.out_strides = tuple(patch_size * 2**level for level in levels)
        self.feature_info = [
            {
                "num_chs": channels,
                "reduction": stride,
                "module": f"layers.{stage}",
            }
            for stage, channels, stride in zip(
                self.out_indices, self.out_channels, self.out_strides
            )
        ]

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.detector_router = DetectorTypeRouter(
            dim=embed_dim,
            hidden_dim=detector_router_hidden_dim,
            norm_layer=norm_layer,
        )
        self.radiation = RadiationInspiredResidualBlock(
            dim=embed_dim,
            norm_layer=norm_layer,
            layer_scale_init=radiation_layer_scale_init,
        )
        self.atmosphere = AtmosphereInspiredResidualBlock(
            dim=embed_dim,
            norm_layer=norm_layer,
            layer_scale_init=atmosphere_layer_scale_init,
            optical_depth_bias_init=atmosphere_optical_depth_bias_init,
        )
        self.tim = ThermalInertiaInspiredResidualBlock(
            dim=embed_dim,
            norm_layer=norm_layer,
            layer_scale_init=tim_layer_scale_init,
        )
        self.layers = nn.ModuleList(
            [
                BasicBlock(
                    index=index,
                    depths=depths,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    drop_path=drop_path_rate,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    norm_layer=norm_layer,
                    patch_merging=index < len(depths) - 1,
                )
                for index in range(len(depths))
            ]
        )
        self.apply(self._init_weights)
        self.detector_router.reset_parameters()
        self.radiation.reset_parameters()
        self.atmosphere.reset_parameters()
        self.tim.reset_parameters()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _physics_front_end(
        self, images: torch.Tensor, return_physics: bool
    ):
        if images.ndim != 4 or images.shape[1] != self.in_chans:
            raise ValueError(
                f"images must be [B,{self.in_chans},H,W], got {tuple(images.shape)}"
            )
        tokens = self.patch_embed(images)
        detector_logits, uncooled_probability = self.detector_router(tokens)
        visible = tokens.new_ones(tokens.shape[:-1] + (1,))
        radiation_tokens, radiation_diagnostics = self.radiation(
            tokens, visible, return_diagnostics=return_physics
        )
        atmosphere_tokens, atmosphere_diagnostics = self.atmosphere(
            radiation_tokens, visible, return_diagnostics=return_physics
        )
        tokens, tim_diagnostics = self.tim(
            atmosphere_tokens,
            visible,
            uncooled_probability.detach(),
            return_diagnostics=return_physics,
        )
        if not return_physics:
            return tokens, {}
        diagnostics = {
            **radiation_diagnostics,
            **atmosphere_diagnostics,
            **tim_diagnostics,
            "detector_logits": detector_logits,
            "uncooled_probability": uncooled_probability,
            "visible": visible,
        }
        return tokens, diagnostics

    def forward_features(
        self,
        images: torch.Tensor,
        return_physics: bool = False,
    ):
        x, diagnostics = self._physics_front_end(images, return_physics)
        outputs = []
        for stage_index, layer in enumerate(self.layers):
            if self.output_mode == "pre_downsample":
                for block in layer.blocks:
                    x = block(x)
                if stage_index in self.out_indices:
                    nchw = x.permute(0, 3, 1, 2).contiguous()
                    outputs.append(nchw if self.channels_first else x)
                if layer.downsample is not None:
                    x = layer.downsample(x)
            else:
                x = layer(x)
                if stage_index in self.out_indices:
                    nchw = x.permute(0, 3, 1, 2).contiguous()
                    outputs.append(nchw if self.channels_first else x)
        result = tuple(outputs)
        if return_physics:
            diagnostics["stage_features"] = result
            return result, diagnostics
        return result

    def forward(self, images: torch.Tensor):
        return self.forward_features(images)

    def forward_global(self, images: torch.Tensor) -> torch.Tensor:
        """Globally pool the deepest returned feature for classification heads."""
        deepest = self.forward_features(images)[-1]
        if self.channels_first:
            return deepest.mean(dim=(-2, -1))
        return deepest.mean(dim=(1, 2))

    def load_pretrained(
        self,
        checkpoint: CheckpointSource,
        require_physics: bool = True,
    ) -> CheckpointLoadReport:
        """Load encoder weights from an AAA_phys pre-training checkpoint."""
        report = load_matching_weights(self, checkpoint)
        if report.loaded_count == 0:
            raise RuntimeError("checkpoint did not load any backbone tensor")
        if require_physics:
            loaded = set(report.loaded_keys)
            absent = [
                key
                for key in self.state_dict()
                if key.startswith(PHYSICS_PREFIXES) and key not in loaded
            ]
            if absent:
                raise RuntimeError(
                    "checkpoint is missing revised physics tensors: "
                    + ", ".join(absent)
                    + "; set require_physics=False only for an intentional "
                    "vanilla-Swin warm start"
                )
        return report


def thermalfm_physics_backbone(**kwargs) -> ThermalFMPhysicsBackbone:
    """Build the Swin-Base architecture used by AAA_phys pre-training."""
    return ThermalFMPhysicsBackbone(
        patch_size=4,
        in_chans=3,
        depths=(2, 2, 18, 2),
        embed_dim=128,
        num_heads=(4, 8, 16, 32),
        window_size=7,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
        **kwargs,
    )
