"""Feature-domain modules for the final detector-aware AAA_phys model.

Radiation, Atmosphere, and TIM operate on dimensionless latent features. The
detector router predicts a sample-level probability from unmasked image
tokens. None of these outputs is a calibrated physical measurement.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectorTypeRouter(nn.Module):
    """Predict the probability that a sample comes from an uncooled sensor.

    The router pools unmasked PatchEmbedding tokens, so training and downstream
    inference both use only one image. Labels supervise the router during
    pre-training, but are never inputs to the forward path.
    """

    def __init__(self, dim: int, hidden_dim: int = 0, norm_layer=nn.LayerNorm):
        super().__init__()
        hidden_dim = hidden_dim or max(dim // 2, 16)
        self.norm = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must be [B,H,W,C], got shape {tuple(tokens.shape)}"
            )
        pooled = self.norm(tokens).mean(dim=(1, 2))
        logits = self.mlp(pooled).squeeze(-1)
        return logits, torch.sigmoid(logits)


class RadiationInspiredResidualBlock(nn.Module):
    """Graybody-inspired emission/background mixing for visible tokens.

    The dimensionless latent interaction borrows the opaque-graybody form

    ``e * U(T_object) + (1 - e) * U(T_background)``.

    ``temperature_like**4`` supplies the emission-like response, while a
    visible-token scene summary supplies the effective background-like
    response. None of these tensors is a calibrated physical quantity.

    Mask convention:
      * ``mask == 1``: masked token
      * ``mask == 0``: visible token

    The caller passes ``visible = 1 - mask`` with shape ``[B, H, W, 1]``.
    Consequently, the residual is exactly zero at every masked location.
    """

    def __init__(
        self,
        dim: int,
        norm_layer=nn.LayerNorm,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if layer_scale_init < 0:
            raise ValueError("layer_scale_init must be non-negative")

        self.norm = norm_layer(dim)
        self.emissivity_head = nn.Linear(dim, dim)
        self.temperature_head = nn.Linear(dim, dim)
        self.background_head = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init))

    def reset_parameters(self) -> None:
        """Initialize a trainable but near-identity residual block."""
        for module in (
            self.emissivity_head,
            self.temperature_head,
            self.background_head,
            self.output_proj,
        ):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        visible: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must be [B,H,W,C], got shape {tuple(tokens.shape)}"
            )
        if visible.shape != tokens.shape[:-1] + (1,):
            raise ValueError(
                "visible must be [B,H,W,1] and match tokens; "
                f"got {tuple(visible.shape)} for {tuple(tokens.shape)}"
            )

        visible = visible.to(dtype=tokens.dtype)
        normalized = self.norm(tokens)
        emissivity_like = torch.sigmoid(
            self.emissivity_head(normalized)
        )
        temperature_like = torch.sigmoid(
            self.temperature_head(normalized)
        )

        # Estimate one effective scene-background response per sample using
        # visible tokens only. Mask tokens therefore cannot leak into the
        # global context used by the physical block.
        visible_count = visible.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        background_context = (
            (normalized * visible).sum(dim=(1, 2), keepdim=True)
            / visible_count
        )
        background_like = torch.sigmoid(
            self.background_head(background_context)
        )

        emission_response = emissivity_like * temperature_like.pow(4)
        reflected_background_response = (
            (1.0 - emissivity_like) * background_like
        )
        radiation_response = (
            emission_response + reflected_background_response
        )

        projected = torch.tanh(self.output_proj(radiation_response))
        residual = visible * self.gamma.view(1, 1, 1, -1) * projected
        output = tokens + residual

        if not return_diagnostics:
            return output, {}

        diagnostics = {
            "emissivity_like": emissivity_like,
            "temperature_like": temperature_like,
            "background_context": background_context,
            "background_like": background_like,
            "emission_response": emission_response,
            "reflected_background_response": reflected_background_response,
            "radiation_response": radiation_response,
            "radiation_residual": residual,
            "radiation_layer_scale": self.gamma,
        }
        return output, diagnostics


class AtmosphereInspiredResidualBlock(nn.Module):
    """Homogeneous-atmosphere-inspired residual for visible tokens.

    A non-negative optical-depth-like scalar is predicted for each token and
    mapped to ``(0, 1]`` through a Beer--Lambert-inspired negative
    exponential. A single bounded atmospheric-source-like feature vector is
    derived from the visible-token context of each sample. The full local path
    response is the complementary term ``(1 - transmission_like) *
    atmospheric_source_like``.

    All internal tensors are dimensionless latent activations. In particular,
    they are not calibrated atmospheric transmission, optical depth, air
    temperature, or path radiance.

    Mask convention:
      * ``mask == 1``: masked token
      * ``mask == 0``: visible token

    The caller passes ``visible = 1 - mask`` with shape ``[B, H, W, 1]``.
    Consequently, the residual is exactly zero at every masked location and
    masked tokens cannot enter the global atmospheric context.
    """

    def __init__(
        self,
        dim: int,
        norm_layer=nn.LayerNorm,
        layer_scale_init: float = 1.0e-4,
        optical_depth_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        if layer_scale_init < 0:
            raise ValueError("layer_scale_init must be non-negative")

        self.optical_depth_bias_init = optical_depth_bias_init
        self.norm = norm_layer(dim)
        self.optical_depth_head = nn.Linear(dim, 1)
        self.atmospheric_source_head = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init))

    def reset_parameters(self) -> None:
        """Initialize a mild-transmission, trainable residual block."""
        # Start with one spatially uniform, relatively high transmission. The
        # zero weight remains trainable because output_proj and gamma are both
        # nonzero at initialization.
        nn.init.zeros_(self.optical_depth_head.weight)
        nn.init.constant_(
            self.optical_depth_head.bias,
            self.optical_depth_bias_init,
        )
        for module in (
            self.atmospheric_source_head,
            self.output_proj,
        ):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        visible: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must be [B,H,W,C], got shape {tuple(tokens.shape)}"
            )
        if visible.shape != tokens.shape[:-1] + (1,):
            raise ValueError(
                "visible must be [B,H,W,1] and match tokens; "
                f"got {tuple(visible.shape)} for {tuple(tokens.shape)}"
            )

        visible = visible.to(dtype=tokens.dtype)
        normalized = self.norm(tokens)

        # One scalar gate per token is shared by all latent feature channels;
        # feature channels are not treated as wavelength channels.
        optical_depth_like = F.softplus(
            self.optical_depth_head(normalized)
        )
        transmission_like = torch.exp(-optical_depth_like)

        visible_count = visible.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        atmospheric_source_context = (
            (normalized * visible).sum(dim=(1, 2), keepdim=True)
            / visible_count
        )
        atmospheric_source_like = torch.tanh(
            self.atmospheric_source_head(atmospheric_source_context)
        )

        # Difference form guarantees a zero atmospheric change as
        # transmission_like approaches one.
        atmospheric_change = (
            (1.0 - transmission_like)
            * (atmospheric_source_like - normalized)
        )
        projected = torch.tanh(self.output_proj(atmospheric_change))
        residual = visible * self.gamma.view(1, 1, 1, -1) * projected
        output = tokens + residual

        if not return_diagnostics:
            return output, {}

        transmitted_response = transmission_like * normalized
        atmospheric_path_response = (
            (1.0 - transmission_like) * atmospheric_source_like
        )
        mixed_atmospheric_response = (
            transmitted_response + atmospheric_path_response
        )

        diagnostics = {
            "optical_depth_like": optical_depth_like,
            "transmission_like": transmission_like,
            "atmospheric_source_context": atmospheric_source_context,
            "atmospheric_source_like": atmospheric_source_like,
            "transmitted_response": transmitted_response,
            "atmospheric_path_response": atmospheric_path_response,
            "mixed_atmospheric_response": mixed_atmospheric_response,
            "atmospheric_change": atmospheric_change,
            "atmosphere_residual": residual,
            "atmosphere_layer_scale": self.gamma,
        }
        return output, diagnostics


class ThermalInertiaInspiredResidualBlock(nn.Module):
    """Single-frame, five-time-constant thermal-inertia approximation.

    Truncating the first-order response integral at ``t - 5*tau`` leaves the
    fixed coefficient ``eta = exp(-5)``. A single processed image cannot
    identify ``tau``, heat capacity, conductance, or a past temperature, so
    this block learns current-like and history-like latent states instead of
    claiming calibrated detector dynamics.
    """

    def __init__(
        self,
        dim: int,
        norm_layer=nn.LayerNorm,
        layer_scale_init: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if layer_scale_init < 0:
            raise ValueError("layer_scale_init must be non-negative")
        self.norm = norm_layer(dim)
        self.current_head = nn.Linear(dim, dim)
        self.history_head = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.register_buffer("five_tau_eta", torch.exp(torch.tensor(-5.0)))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.norm.bias)
        nn.init.ones_(self.norm.weight)
        for module in (self.current_head, self.history_head, self.output_proj):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        visible: torch.Tensor,
        uncooled_probability: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must be [B,H,W,C], got shape {tuple(tokens.shape)}"
            )
        if visible.shape != tokens.shape[:-1] + (1,):
            raise ValueError("visible must be [B,H,W,1] and match tokens")
        if uncooled_probability.shape != (tokens.shape[0],):
            raise ValueError("uncooled_probability must have shape [B]")

        visible = visible.to(dtype=tokens.dtype)
        normalized = self.norm(tokens)
        visible_count = visible.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        context = (normalized * visible).sum(dim=(1, 2), keepdim=True) / visible_count
        current_like = torch.tanh(self.current_head(normalized))
        history_like = torch.tanh(self.history_head(context))
        eta = self.five_tau_eta.to(dtype=tokens.dtype)
        five_tau_state = eta * history_like + (1.0 - eta) * current_like
        raw_residual = (
            visible
            * self.gamma.view(1, 1, 1, -1)
            * torch.tanh(self.output_proj(five_tau_state))
        )
        gate = uncooled_probability.to(dtype=tokens.dtype).view(-1, 1, 1, 1)
        effective_residual = gate * raw_residual
        output = tokens + effective_residual

        if not return_diagnostics:
            return output, {}
        return output, {
            "tim_current_like": current_like,
            "tim_history_like": history_like,
            "tim_five_tau_state": five_tau_state,
            "tim_five_tau_eta": eta,
            "tim_raw_residual": raw_residual,
            "tim_effective_residual": effective_residual,
            "tim_gate": gate,
            "tim_layer_scale": self.gamma,
        }
