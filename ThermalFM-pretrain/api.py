"""Factories retain the parameter names and shapes of full checkpoint-299."""

from functools import partial

import torch.nn as nn

from .backbone import ThermalFMPhysicsBackbone
from .model import SwinMAEFeatureSerial

PIPELINE_VERSION = "aaa_phys_detector_gated_tim5_v1"
FULL_CONFIG = dict(
    img_size=224, patch_size=4, in_chans=3, embed_dim=128,
    decoder_embed_dim=1024, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
    window_size=7, qkv_bias=True, mlp_ratio=4.0, drop_path_rate=0.4,
    drop_rate=0.0, attn_drop_rate=0.0, patch_norm=True,
    mask_ratio=0.75, mask_window_size=4, norm_pix_loss=False,
    detector_router_hidden_dim=0, detector_loss_weight=0.01, detector_pos_weight=1.0,
    radiation_layer_scale_init=1e-4, atmosphere_layer_scale_init=1e-4,
    atmosphere_optical_depth_bias_init=-2.0, tim_layer_scale_init=1e-4,
    frequency_loss_weight=0.05, frequency_cutoff_ratio=0.25, frequency_order=2,
)
TINY_CONFIG = dict(
    img_size=64, embed_dim=16, decoder_embed_dim=128, depths=(1, 1, 1, 1),
    num_heads=(1, 2, 4, 8), window_size=2, mask_window_size=2, drop_path_rate=0.0,
)


def model_config(tiny=False, **overrides):
    return {**FULL_CONFIG, **(TINY_CONFIG if tiny else {}), **overrides}


def create_pretrainer(**config):
    config = model_config(**config)
    model = SwinMAEFeatureSerial(
        **config, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    model.config = config
    return model


def create_backbone(checkpoint=None, output_mode="pre_downsample", **config):
    config = model_config(**config)
    encoder_keys = (
        "patch_size", "in_chans", "embed_dim", "depths", "num_heads", "window_size",
        "qkv_bias", "mlp_ratio", "drop_path_rate", "drop_rate", "attn_drop_rate",
        "patch_norm", "detector_router_hidden_dim", "radiation_layer_scale_init",
        "atmosphere_layer_scale_init", "atmosphere_optical_depth_bias_init",
        "tim_layer_scale_init",
    )
    model = ThermalFMPhysicsBackbone(
        **{key: config[key] for key in encoder_keys},
        norm_layer=partial(nn.LayerNorm, eps=1e-6), output_mode=output_mode,
    )
    model.config = config
    if checkpoint is not None:
        report = model.load_pretrained(checkpoint, require_physics=True)
        if report.missing_keys or report.shape_mismatches:
            raise RuntimeError(report.summary())
        model.load_report = report
    return model


class ThermalClassifier(nn.Module):
    """Dense stage-4 GAP + linear head, with optional fully frozen backbone.

    The original gate remains detached: classification-only fine-tuning does
    not train the router's parameters through TIM. DDP must allow unused
    parameters when the backbone is unfrozen.
    """

    def __init__(self, backbone, num_classes, frozen=True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.backbone = backbone
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.head = nn.Linear(backbone.out_channels[-1], num_classes)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, images):
        return self.head(self.backbone.forward_global(images))
