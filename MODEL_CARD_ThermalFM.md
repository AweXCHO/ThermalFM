# Model Card: ThermalFM

## 1. Model identity

| Field | Value |
|---|---|
| Suggested name | ThermalFM |
| Full pretraining model parameters | **148,833,634 (148.83M)**, including encoder, physics front end, router, and reconstruction decoder |
| Downstream backbone parameters | **86,899,642 (86.90M)**, excluding the reconstruction decoder |
| Physics front end and router parameters | 158,466, already included in the totals above |
| Full state dictionary | 709 tensors, including parameters and buffers; not the number of parameter tensors |
| Checkpoint file size | 1,784,746,858 bytes, approximately 1.66 GiB; includes optimizer, scaler, and arguments, not just backbone weights |

## 2. Design and architecture

### 2.1 Encoding and reconstruction paths

```text
Single normalized thermal image
        │
        ▼
Patch Embedding ──→ Router (complete, pre-mask tokens) ──→ p_u
        │                                                  │
        ▼                                                  │ stopgrad
MAE window masking (omitted for downstream features)        │
        ▼                                                  │
Radiation residual                                         │
        ▼                                                  │
Atmosphere residual                                        │
        ▼                                                  │
5τ TIM residual ←──────────────────────────────────────────┘
        ▼
4-stage Swin Encoder
        ├──→ MAE Decoder: pretraining / reconstruction diagnostics
        └──→ Multiscale / global features: separate downstream head
```

| Configuration | Value |
|---|---|
| Encoder | Hierarchical Transformer with a Swin-Base configuration |
| Training input | `3 × 224 × 224` |
| Patch size / attention window size | 4 / 7 |
| Base embedding dimension | 128 |
| Stage depths | `(2, 2, 18, 2)` |
| Attention heads | `(4, 8, 16, 32)` |
| MLP ratio | 4 |
| DropPath rate | 0.4; use `eval()` for deterministic feature extraction |
| LayerNorm epsilon | `1e-6` |
| Deepest feature dimension / decoder embedding dimension | 1024 / 1024 |
| MAE mask ratio / masking window | 0.75 / `4 × 4` patch tokens |
| Router | LayerNorm → spatial mean pooling → Linear(128, 64) → GELU → Linear(64, 1) → Sigmoid |

### 2.2 Interpretation of the physics-inspired modules

- **Radiation:** a learnable residual inspired by graybody emission and background mixing.
- **Atmosphere:** a residual inspired by Beer–Lambert-style transmission and atmospheric path response.
- **TIM:** a single-frame residual inspired by a response truncated at 5τ. A fixed coefficient, `η = exp(-5)`, combines current-image latent variables and visible-region context as `H_5τ = ηH_0 + (1−η)Q`.
- **Detector router:** predicts `p_u` from the image, with labels `0=cooled` and `1=uncooled`. The gated TIM output is `F_out = F_a + stopgrad(p_u) × Δ_TIM`.

During MAE training, the physics branches inject residuals only into visible tokens. The router reads complete patch tokens before masking and therefore has access to domain cues from the complete single image. Labels supervise the BCE objective but are not gate inputs. The reconstruction path stops gradients through the gate, but the router and main path still share Patch Embedding; they are not two entirely independent networks.

All temperature-like, emissivity-like, transmission-like, and history-like quantities are **dimensionless latent variables**, not measured temperature, emissivity, distance, heat capacity, thermal conductance, or time constants. 

## 3. Inputs, outputs, and operating conditions

### 3.1 Inputs

The expected input is a white-hot thermal-infrared intensity image, mapped to 8-bit grayscale and replicated into three identical channels. The model accepts floating-point tensors of shape `[B, 3, H, W]`.

Training augmentation uses `RandomResizedCrop(224, scale=(0.2, 1.0), bicubic)` and random horizontal flipping. The standard reconstruction/classification diagnostic protocol in this card resizes the shorter side to 256 and takes a 224 center crop. Direct square resizing, uncropped resizing, and 448-pixel inputs are different protocols and must be reported separately.

Do not call `Image.convert("L")` on raw 16-bit or floating-point thermal images and assume that their dynamic range has been correctly mapped: this may saturate the input to white. High-bit-depth inputs require an explicit sensor adaptation, invalid-pixel handling, and bit-depth mapping policy, followed by model revalidation. Per-image min–max scaling also changes brightness semantics and must not silently replace the original protocol.

Black-hot polarity, pseudocolor, raw radiometric counts, absolute-temperature maps, and visible-light RGB are not validated standard inputs under this card.

### 3.2 Outputs

The full MAE's standard interface returns `loss, pred, mask`. Here, `pred` contains patch reconstructions in the normalized domain, and `mask=1` marks a masked patch. Predictions must be inverse-normalized to obtain intensity-domain values.

The standalone downstream backbone omits the MAE mask, decoder, and reconstruction losses. With `output_mode="pre_downsample"`, a 224-pixel input produces:

| Level | Feature shape | Stride |
|---|---|---:|
| Stage 1 | `[B, 128, 56, 56]` | 4 |
| Stage 2 | `[B, 256, 28, 28]` | 8 |
| Stage 3 | `[B, 512, 14, 14]` | 16 |
| Stage 4 | `[B, 1024, 7, 7]` | 32 |

Global average pooling of the deepest feature gives a 1024-dimensional vector. Optional diagnostics include `uncooled_probability` with shape `[B]` and physics residuals. The alternative `post_stage` mode returns channels/strides `(256, 512, 1024, 1024)` / `(8, 16, 32, 32)`; it must not be confused with the conventional pyramid above.

The frequency-domain component is a **training objective**, not an additional frequency-feature branch at inference time.
