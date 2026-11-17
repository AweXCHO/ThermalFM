"""Swin-MAE with serial latent physics blocks and a single-image router.

The final path is PatchEmbedding -> mask -> Radiation -> Atmosphere -> TIM.
A detector-type router reads the unmasked PatchEmbedding tokens before MAE
masking and supplies the TIM gate at both training and inference.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .swin import BasicBlock, BasicBlockUp, PatchEmbedding, PatchExpanding
from .pos_embed import get_2d_sincos_pos_embed

from .physics import (
    AtmosphereInspiredResidualBlock,
    DetectorTypeRouter,
    RadiationInspiredResidualBlock,
    ThermalInertiaInspiredResidualBlock,
)


class SwinMAEFeatureSerial(nn.Module):
    """MAE whose main token path contains serial physics-inspired blocks."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        in_chans: int = 3,
        decoder_embed_dim: int = 512,
        norm_pix_loss: bool = False,
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
        mask_window_size: int = 4,
        detector_router_hidden_dim: int = 0,
        detector_loss_weight: float = 0.01,
        detector_pos_weight: float = 1.0,
        radiation_layer_scale_init: float = 1.0e-4,
        atmosphere_layer_scale_init: float = 1.0e-4,
        atmosphere_optical_depth_bias_init: float = -2.0,
        tim_layer_scale_init: float = 1.0e-4,
        frequency_loss_weight: float = 0.05,
        frequency_cutoff_ratio: float = 0.25,
        frequency_order: int = 2,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if frequency_loss_weight < 0:
            raise ValueError("frequency_loss_weight must be non-negative")
        if detector_loss_weight < 0:
            raise ValueError("detector_loss_weight must be non-negative")
        if detector_pos_weight <= 0:
            raise ValueError("detector_pos_weight must be positive")
        if not 0.0 < frequency_cutoff_ratio < 1.0:
            raise ValueError("frequency_cutoff_ratio must lie in (0, 1)")
        if frequency_order <= 0:
            raise ValueError("frequency_order must be positive")
        expected_decoder_dim = embed_dim * 2 ** (len(depths) - 1)
        if decoder_embed_dim != expected_decoder_dim:
            raise ValueError(
                "decoder_embed_dim must equal the deepest encoder dimension "
                f"({expected_decoder_dim}), got {decoder_embed_dim}"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_window_size = mask_window_size
        self.num_patches = (img_size // patch_size) ** 2
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
        self.in_chans = in_chans
        self.frequency_loss_weight = float(frequency_loss_weight)
        self.detector_loss_weight = float(detector_loss_weight)
        self.detector_pos_weight = float(detector_pos_weight)

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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

        self.layers = self.build_layers()
        self.first_patch_expanding = PatchExpanding(
            dim=decoder_embed_dim, norm_layer=norm_layer
        )
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim // 8,
            patch_size**2 * in_chans,
            bias=True,
        )

        self.register_buffer(
            "low_frequency_weight",
            self._generate_low_frequency_weight(
                frequency_cutoff_ratio,
                frequency_order,
                img_size,
                img_size,
            ),
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        nn.init.normal_(self.mask_token, std=0.02)
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

    def build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for index in range(self.num_layers):
            layers.append(
                BasicBlock(
                    index=index,
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
                    patch_merging=index < self.num_layers - 1,
                )
            )
        return layers

    def build_layers_up(self) -> nn.ModuleList:
        layers_up = nn.ModuleList()
        for index in range(self.num_layers - 1):
            layers_up.append(
                BasicBlockUp(
                    index=index,
                    depths=self.depths,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=self.drop_path,
                    window_size=self.window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    drop_rate=self.drop_rate,
                    attn_drop_rate=self.attn_drop_rate,
                    patch_expanding=index < self.num_layers - 2,
                    norm_layer=self.norm_layer,
                )
            )
        return layers_up

    def _generate_window_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return 1 for masked tokens and 0 for visible tokens."""
        r = self.mask_window_size
        if height % r != 0 or width % r != 0:
            raise ValueError(
                f"token grid {height}x{width} must be divisible by "
                f"mask_window_size={r}"
            )

        block_h, block_w = height // r, width // r
        block_count = block_h * block_w
        keep_count = int(block_count * (1.0 - self.mask_ratio))
        noise = torch.rand(batch_size, block_count, device=device)
        keep = noise.argsort(dim=1)[:, :keep_count]

        # Start masked (1), then mark kept/visible windows as 0.
        block_mask = torch.ones(batch_size, block_count, device=device)
        block_mask.scatter_(1, keep, 0.0)
        block_mask = block_mask.view(batch_size, block_h, block_w)
        return block_mask.repeat_interleave(r, dim=1).repeat_interleave(
            r, dim=2
        )

    def _apply_mask(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, height, width, _ = tokens.shape
        mask_2d = self._generate_window_mask(
            batch_size, height, width, tokens.device
        )
        mask_bool = mask_2d.unsqueeze(-1).bool()
        mask_token = self.mask_token.view(1, 1, 1, -1).to(tokens.dtype)
        masked_tokens = torch.where(mask_bool, mask_token, tokens)
        return masked_tokens, mask_2d.flatten(1)

    def forward_encoder(
        self,
        images: torch.Tensor,
        return_physics: bool = False,
        return_detector: bool = False,
    ):
        unmasked_tokens = self.patch_embed(images)
        detector_logits, uncooled_probability = self.detector_router(
            unmasked_tokens
        )
        base_tokens, mask = self._apply_mask(unmasked_tokens)

        height, width = base_tokens.shape[1:3]
        # mask=1 means masked, so 1-mask is exactly the visible region.
        visible = (1.0 - mask).view(mask.shape[0], height, width, 1)
        radiation_tokens, radiation_diagnostics = self.radiation(
            base_tokens,
            visible,
            return_diagnostics=return_physics,
        )
        atmosphere_tokens, atmosphere_diagnostics = self.atmosphere(
            radiation_tokens,
            visible,
            return_diagnostics=return_physics,
        )
        # Reconstruction is not allowed to turn the detector probability into
        # an unconstrained shortcut. The router learns from its labeled BCE;
        # TIM and the rest of the encoder still receive reconstruction grads.
        tim_tokens, tim_diagnostics = self.tim(
            atmosphere_tokens,
            visible,
            uncooled_probability.detach(),
            return_diagnostics=return_physics,
        )

        # The three blocks generate dependency-preserving serial residuals:
        #   radiation_tokens  = base_tokens + delta_r
        #   atmosphere_tokens = radiation_tokens + delta_a
        #   tim_tokens        = atmosphere_tokens + p_u * delta_tim
        # Hence tim_tokens is the exact additive fusion used below.
        fused_tokens = tim_tokens

        for layer in self.layers:
            fused_tokens = layer(fused_tokens)

        if return_physics:
            radiation_residual = radiation_diagnostics["radiation_residual"]
            atmosphere_residual = atmosphere_diagnostics[
                "atmosphere_residual"
            ]
            tim_residual = tim_diagnostics["tim_effective_residual"]
            tim_raw_residual = tim_diagnostics["tim_raw_residual"]
            additive_fusion = (
                base_tokens
                + radiation_residual
                + atmosphere_residual
                + tim_residual
            )
            base_norm = (
                (base_tokens * visible).flatten(1).norm(dim=1)
                .clamp_min(torch.finfo(base_tokens.dtype).eps)
            )
            radiation_norm = radiation_residual.flatten(1).norm(dim=1)
            atmosphere_norm = atmosphere_residual.flatten(1).norm(dim=1)
            tim_norm = tim_residual.flatten(1).norm(dim=1)
            tim_raw_norm = tim_raw_residual.flatten(1).norm(dim=1)
            physics_diagnostics = {
                **radiation_diagnostics,
                **atmosphere_diagnostics,
                **tim_diagnostics,
                "detector_logits": detector_logits,
                "uncooled_probability": uncooled_probability,
                "additive_fusion_error": (
                    tim_tokens - additive_fusion
                ).abs().amax(),
                "radiation_residual_ratio": radiation_norm / base_norm,
                "atmosphere_residual_ratio": atmosphere_norm / base_norm,
                "tim_raw_residual_ratio": tim_raw_norm / base_norm,
                "tim_effective_residual_ratio": tim_norm / base_norm,
                "atmosphere_to_radiation_residual_ratio": (
                    atmosphere_norm
                    / radiation_norm.clamp_min(
                        torch.finfo(radiation_norm.dtype).eps
                    )
                ),
                "visible": visible,
                "mask": mask,
            }
            return fused_tokens, mask, physics_diagnostics
        if return_detector:
            return fused_tokens, mask, detector_logits, uncooled_probability
        return fused_tokens, mask

    def forward_decoder(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.first_patch_expanding(latent)
        for layer in self.layers_up:
            x = layer(x)
        x = self.norm_up(x)
        x = rearrange(x, "B H W C -> B (H W) C")
        return self.decoder_pred(x)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        if images.shape[2] != images.shape[3] or images.shape[2] % p != 0:
            raise ValueError("patchify expects square images divisible by patch_size")
        h = w = images.shape[2] // p
        x = images.reshape(images.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(
            images.shape[0], h * w, p**2 * self.in_chans
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        if h * w != patches.shape[1]:
            raise ValueError("number of patches must form a square grid")
        x = patches.reshape(
            patches.shape[0], h, w, p, p, self.in_chans
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            patches.shape[0], self.in_chans, h * p, w * p
        )

    def forward_loss(
        self,
        images: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        return_components: bool = False,
    ):
        # The primary MAE target is always the unfiltered observation. No
        # frequency is removed from the supervision target.
        raw_target = self.patchify(images)
        target = raw_target
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            patch_std = torch.sqrt(var + 1.0e-6)
            target = (target - mean) / patch_std
            pred_for_spectrum = pred * patch_std + mean
        else:
            pred_for_spectrum = pred

        patch_loss = (pred - target).pow(2).mean(dim=-1)
        pixel_loss = (
            (patch_loss * mask).sum() / mask.sum().clamp_min(1.0)
        )

        if self.frequency_loss_weight > 0.0:
            predicted_images = self.unpatchify(pred_for_spectrum)
            token_height = images.shape[-2] // self.patch_size
            token_width = images.shape[-1] // self.patch_size
            pixel_mask = mask.view(
                mask.shape[0], token_height, token_width
            )
            pixel_mask = pixel_mask.repeat_interleave(
                self.patch_size, dim=1
            ).repeat_interleave(self.patch_size, dim=2).unsqueeze(1)
            pixel_mask = pixel_mask.to(dtype=images.dtype)

            # Visible pixels remain equal to the observation; only predicted
            # masked regions contribute reconstruction error. Keep the error
            # in the same normalized RGB domain as the primary pixel loss so
            # that a small frequency_loss_weight has a meaningful scale.
            composite_images = (
                images * (1.0 - pixel_mask)
                + predicted_images * pixel_mask
            )
            reconstruction_error = composite_images - images

            error_spectrum = torch.fft.fftshift(
                torch.fft.fft2(
                    reconstruction_error.float(),
                    dim=(-2, -1),
                    norm="ortho",
                ),
                dim=(-2, -1),
            )
            spectral_energy = error_spectrum.abs().square()
            frequency_weight = self.low_frequency_weight
            weight_sum = frequency_weight.sum().clamp_min(
                torch.finfo(frequency_weight.dtype).eps
            )
            low_frequency_loss = (
                (spectral_energy * frequency_weight)
                .sum(dim=(1, 2, 3))
                .div(weight_sum * self.in_chans)
                .div(mask.float().mean(dim=1).clamp_min(1.0e-6))
                .mean()
            )
        else:
            low_frequency_loss = pixel_loss.new_zeros(())

        total_loss = (
            pixel_loss
            + self.frequency_loss_weight * low_frequency_loss
        )
        if not return_components:
            return total_loss

        components = {
            "pixel_reconstruction_loss": pixel_loss,
            "low_frequency_residual_loss": low_frequency_loss,
            "frequency_loss_weight": total_loss.new_tensor(
                self.frequency_loss_weight
            ),
            "total_reconstruction_loss": total_loss,
        }
        return total_loss, components

    def forward(
        self,
        images: torch.Tensor,
        detector_labels: torch.Tensor | None = None,
        return_physics: bool = False,
        return_loss_components: bool = False,
    ):
        if return_physics and return_loss_components:
            raise ValueError(
                "return_physics already includes loss components; do not set "
                "return_loss_components at the same time"
            )
        if return_physics:
            latent, mask, diagnostics = self.forward_encoder(
                images, return_physics=True
            )
            detector_logits = diagnostics["detector_logits"]
            uncooled_probability = diagnostics["uncooled_probability"]
        else:
            (
                latent,
                mask,
                detector_logits,
                uncooled_probability,
            ) = self.forward_encoder(images, return_detector=True)

        pred = self.forward_decoder(latent)
        if return_physics or return_loss_components:
            reconstruction_loss, loss_components = self.forward_loss(
                images,
                pred,
                mask,
                return_components=True,
            )
        else:
            reconstruction_loss = self.forward_loss(images, pred, mask)

        if detector_labels is not None:
            detector_labels = detector_labels.to(
                device=detector_logits.device,
                dtype=detector_logits.dtype,
            ).reshape(-1)
            if detector_labels.shape != detector_logits.shape:
                raise ValueError(
                    "detector_labels must contain one binary label per image"
                )
            if not torch.all((detector_labels == 0) | (detector_labels == 1)):
                raise ValueError("detector_labels must contain only 0 or 1")
            positive_weight = detector_logits.new_tensor(
                self.detector_pos_weight
            )
            detector_loss = nn.functional.binary_cross_entropy_with_logits(
                detector_logits,
                detector_labels,
                pos_weight=positive_weight,
            )
            detector_accuracy = (
                (uncooled_probability >= 0.5) == detector_labels.bool()
            ).float().mean()
        else:
            detector_loss = reconstruction_loss.new_zeros(())
            detector_accuracy = reconstruction_loss.new_zeros(())

        loss = reconstruction_loss + self.detector_loss_weight * detector_loss
        if return_physics or return_loss_components:
            loss_components.update(
                {
                    "detector_classification_loss": detector_loss,
                    "detector_loss_weight": loss.new_tensor(
                        self.detector_loss_weight
                    ),
                    "detector_accuracy": detector_accuracy,
                    "uncooled_probability_mean": uncooled_probability.mean(),
                    "total_loss": loss,
                }
            )
            if detector_labels is not None:
                cooled = detector_labels == 0
                uncooled = detector_labels == 1
                loss_components["cooled_count"] = cooled.sum()
                loss_components["uncooled_count"] = uncooled.sum()
                loss_components["cooled_probability_sum"] = (
                    uncooled_probability[cooled].sum()
                )
                loss_components["uncooled_probability_sum"] = (
                    uncooled_probability[uncooled].sum()
                )
            if return_physics:
                diagnostics.update(loss_components)
        if return_physics:
            return loss, pred, mask, diagnostics
        if return_loss_components:
            return loss, pred, mask, loss_components
        return loss, pred, mask

    @staticmethod
    def _generate_low_frequency_weight(
        cutoff_ratio: float,
        order: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        rows = torch.arange(height, dtype=torch.float32).view(height, 1)
        cols = torch.arange(width, dtype=torch.float32).view(1, width)
        distance = torch.sqrt(
            (rows - height // 2).pow(2) + (cols - width // 2).pow(2)
        )
        normalized_distance = distance / distance.max().clamp_min(1.0)
        low_frequency_weight = 1.0 / (
            1.0
            + (normalized_distance / cutoff_ratio).pow(2 * order)
        )
        # The primary pixel loss already supervises the global DC component.
        # Excluding it prevents the auxiliary term from collapsing into a
        # per-image brightness-bias penalty instead of coarse structure.
        low_frequency_weight[height // 2, width // 2] = 0.0
        return low_frequency_weight.unsqueeze(0).unsqueeze(0)
