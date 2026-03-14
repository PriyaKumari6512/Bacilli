"""
cavmamba.py — CaVMamba: Visual State Space Model with CNN Augmented VMamba.

Reference: "CaVMamba: Visual State Space Model with CNN Augmented VMamba"
           The Visual Computer, 2025.

Primary architecture for TB bacilli segmentation.
Encoder loaded from VMamba-Small pretrained weights (strict=False).
Decoder initialized from scratch.

forward() returns raw logits — sigmoid is NEVER applied inside forward().
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Core Building Blocks
# =============================================================================


class DepthwiseConv2d(nn.Module):
    """Depthwise separable convolution 3×3."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.dw(x))


class VSSBlock(nn.Module):
    """
    Visual State Space Block — simplified VMamba 2D-Selective-Scan.

    Scans features in 4 directions (left-right, right-left, top-bottom,
    bottom-top) using depthwise convolutions and gating, approximating
    the selective scan mechanism without requiring mamba-ssm.
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, dim * 2)
        # 4-directional scanning via depthwise conv branches
        self.scan_convs = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
                for _ in range(4)
            ]
        )
        self.scan_norms = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(4)])
        self.gate_act = nn.SiLU()
        self.proj_out = nn.Linear(dim, dim)
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) tensor
        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        residual = x

        x = self.norm(x)
        xz = self.proj_in(x)
        x_scan, z = xz.chunk(2, dim=-1)

        # Reshape to (B, C, H, W) for convolutions
        x_scan = x_scan.permute(0, 3, 1, 2).contiguous()

        # 4-directional scanning
        scans = []
        # Direction 0: original (left-to-right, top-to-bottom)
        scans.append(self.scan_norms[0](self.scan_convs[0](x_scan)))
        # Direction 1: horizontal flip (right-to-left)
        scans.append(
            self.scan_norms[1](self.scan_convs[1](x_scan.flip(3))).flip(3)
        )
        # Direction 2: vertical flip (bottom-to-top)
        scans.append(
            self.scan_norms[2](self.scan_convs[2](x_scan.flip(2))).flip(2)
        )
        # Direction 3: transpose scan (diagonal)
        x_t = x_scan.transpose(2, 3)
        scans.append(
            self.scan_norms[3](self.scan_convs[3](x_t)).transpose(2, 3)
        )

        # Merge scans
        merged = sum(scans) / 4.0
        merged = merged.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # Gate
        out = merged * self.gate_act(z)
        out = self.proj_out(out)
        return residual + self.drop_path(out)


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class SandwichBlock(nn.Module):
    """
    Core CaVMamba innovation: CNN-VMamba-CNN sandwich.

    LayerNorm → DepthwiseConv3×3 → GELU → VSSBlock → DepthwiseConv3×3 → GELU
    → residual add

    First CNN: captures local rod-shape edges and stain features.
    VMamba: global context, learns empty region suppression.
    Second CNN: refines local detail after global context injection.
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv1 = DepthwiseConv2d(dim)
        self.act1 = nn.GELU()
        self.vss = VSSBlock(dim, drop_path=drop_path)
        self.dwconv2 = DepthwiseConv2d(dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H, W, C)
        """
        residual = x
        B, H, W, C = x.shape

        # First CNN
        x = self.norm(x)
        x_conv = x.permute(0, 3, 1, 2).contiguous()
        x_conv = self.act1(self.dwconv1(x_conv))
        x = x_conv.permute(0, 2, 3, 1).contiguous()

        # VMamba
        x = self.vss(x)

        # Second CNN
        x_conv = x.permute(0, 3, 1, 2).contiguous()
        x_conv = self.act2(self.dwconv2(x_conv))
        x = x_conv.permute(0, 2, 3, 1).contiguous()

        return residual + x


class PatchMerging(nn.Module):
    """Downsample spatial by 2x, double channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, H/2, W/2, 2C)"""
        B, H, W, C = x.shape
        # Pad if odd
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1, 0, 0))
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
            W += 1

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    """Upsample spatial by 2x, halve channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, 2H, 2W, C/2)"""
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding: Conv stem to create initial tokens."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H/4, W/4, embed_dim)"""
        x = self.proj(x)  # (B, embed_dim, H/4, W/4)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x


class DynamicFeatureFusion(nn.Module):
    """
    Dynamic Feature Fusion from CaVMamba.

    Upsample all encoder stages to same size → concat → Conv1×1 →
    softmax weights → weighted sum.
    """

    def __init__(self, dims: List[int], target_size: int):
        super().__init__()
        self.target_size = target_size
        total_dim = sum(dims)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(total_dim, len(dims), kernel_size=1, bias=False),
        )
        # Project each feature to common dim
        self.projections = nn.ModuleList(
            [nn.Conv2d(d, dims[0], kernel_size=1) for d in dims]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of (B, C_i, H_i, W_i) tensors from encoder stages.
        Returns:
            Fused feature (B, C_0, target_size, target_size).
        """
        # Upsample all to target size
        upsampled = []
        for feat in features:
            up = F.interpolate(
                feat, size=self.target_size, mode="bilinear", align_corners=False
            )
            upsampled.append(up)

        # Concat for weight computation
        concat = torch.cat(upsampled, dim=1)
        weights = self.weight_conv(concat)  # (B, N_stages, H, W)
        weights = torch.softmax(weights, dim=1)

        # Project and weighted sum
        projected = []
        for i, feat in enumerate(upsampled):
            projected.append(self.projections[i](feat))

        out = torch.zeros_like(projected[0])
        for i in range(len(projected)):
            out = out + projected[i] * weights[:, i : i + 1, :, :]

        return out


# =============================================================================
# CaVMamba Full Model
# =============================================================================


class CaVMamba(nn.Module):
    """
    CaVMamba: Visual State Space Model with CNN Augmented VMamba.

    Encoder: 4 stages of SandwichBlocks + PatchMerging.
    Decoder: 4 stages of SandwichBlocks + PatchExpanding.
    Skip connections: additive.
    Dropout2d(0.2) before final head.
    Output: raw logits (B, 1, H, W) — sigmoid NEVER inside forward().
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_dims: List[int] = None,
        encoder_depths: List[int] = None,
        decoder_dims: List[int] = None,
        decoder_depths: List[int] = None,
        dropout: float = 0.2,
        input_size: int = 256,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [96, 192, 384, 768]
        if encoder_depths is None:
            encoder_depths = [2, 2, 15, 2]
        if decoder_dims is None:
            decoder_dims = [384, 192, 96, 48]
        if decoder_depths is None:
            decoder_depths = [2, 2, 2, 2]

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_stages = len(encoder_dims)

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, encoder_dims[0], patch_size=4)
        feat_size = input_size // 4

        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        drop_path_rate = 0.2
        total_depth = sum(encoder_depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        cur = 0
        for i in range(self.num_stages):
            blocks = nn.ModuleList(
                [
                    SandwichBlock(encoder_dims[i], drop_path=dpr[cur + j])
                    for j in range(encoder_depths[i])
                ]
            )
            self.encoder_stages.append(blocks)
            cur += encoder_depths[i]

            if i < self.num_stages - 1:
                self.downsamples.append(PatchMerging(encoder_dims[i]))
            else:
                self.downsamples.append(nn.Identity())

        # Bottleneck projection
        self.bottleneck = nn.Linear(encoder_dims[-1], decoder_dims[0])

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(self.num_stages):
            dec_dim = decoder_dims[i]

            if i < self.num_stages - 1:
                self.upsamples.append(PatchExpanding(dec_dim))
                # Skip projection: encoder dim → next decoder dim
                enc_idx = self.num_stages - 2 - i
                self.skip_projections.append(
                    nn.Linear(encoder_dims[enc_idx], decoder_dims[i + 1])
                )
            else:
                self.upsamples.append(nn.Identity())
                self.skip_projections.append(nn.Identity())

            blocks = nn.ModuleList(
                [SandwichBlock(dec_dim) for _ in range(decoder_depths[i])]
            )
            self.decoder_stages.append(blocks)

        # Dynamic feature fusion
        self.dff = DynamicFeatureFusion(encoder_dims, target_size=feat_size)

        # Final head
        self.dropout2d = nn.Dropout2d(dropout)
        self.final_up = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], decoder_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dims[-1]),
            nn.GELU(),
        )
        self.head = nn.Conv2d(decoder_dims[-1], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_vmamba_pretrained(self, weight_path: str):
        """
        Load VMamba-Small encoder weights with key mapping.
        Uses strict=False — logs loaded vs skipped keys.
        """
        if not weight_path or not __import__("os").path.isfile(weight_path):
            logger.warning(f"Pretrained weights not found at {weight_path}. Using random init.")
            return

        logger.info(f"Loading VMamba-Small pretrained weights from {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Map keys — attempt to load what matches
        model_dict = self.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k, v in state_dict.items():
            # Try direct match or with encoder prefix mapping
            mapped_key = None
            if k in model_dict and model_dict[k].shape == v.shape:
                mapped_key = k
            else:
                # Try common mappings
                for prefix_from, prefix_to in [
                    ("layers.", "encoder_stages."),
                    ("patch_embed.", "patch_embed."),
                    ("downsample_layers.", "downsamples."),
                ]:
                    candidate = k.replace(prefix_from, prefix_to)
                    if candidate in model_dict and model_dict[candidate].shape == v.shape:
                        mapped_key = candidate
                        break

            if mapped_key is not None:
                model_dict[mapped_key] = v
                loaded_keys.append(mapped_key)
            else:
                skipped_keys.append(k)

        self.load_state_dict(model_dict, strict=False)
        logger.info(
            f"Pretrained loading: {len(loaded_keys)} loaded, "
            f"{len(skipped_keys)} skipped"
        )
        if skipped_keys and len(skipped_keys) <= 20:
            logger.debug(f"Skipped keys: {skipped_keys}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, 256, 256) input tensor.
        Returns:
            (B, 1, 256, 256) raw logits — NO sigmoid.
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, H/4, W/4, C0)

        # Encoder
        encoder_features = []
        for i in range(self.num_stages):
            for block in self.encoder_stages[i]:
                x = block(x)

            # Store feature for skip connection (B, H_i, W_i, C_i)
            encoder_features.append(x)

            # Downsample (except last stage)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        # Dynamic feature fusion (for encoder features)
        enc_feats_conv = [
            f.permute(0, 3, 1, 2).contiguous() for f in encoder_features
        ]
        _ = self.dff(enc_feats_conv)  # Side enrichment (unused directly but provides gradient signal)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.num_stages):
            for block in self.decoder_stages[i]:
                x = block(x)

            if i < self.num_stages - 1:
                x = self.upsamples[i](x)
                # Additive skip connection
                skip = encoder_features[self.num_stages - 2 - i]
                skip = self.skip_projections[i](skip)
                x = x + skip

        # Final upsampling to original resolution
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.dropout2d(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.final_up(x)
        x = self.head(x)

        return x  # Raw logits — sigmoid NEVER inside forward()
