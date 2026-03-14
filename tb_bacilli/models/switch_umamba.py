"""
switch_umamba.py — Switch-UMamba: Dynamic Scanning Vision Mamba UNet.

Reference: "Switch-UMamba: Dynamic Scanning Vision Mamba UNet"
           ScienceDirect, 2025.

Fallback architecture. Trains entirely from scratch — no pretrained weights.
forward() returns raw logits — sigmoid is NEVER applied inside forward().
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Drop Path
# =============================================================================


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


# =============================================================================
# Scan Expert
# =============================================================================


class ScanExpert(nn.Module):
    """
    Single directional scan expert.

    Uses depthwise conv + linear to simulate a directional SSM scan.
    Each expert scans in a different direction.
    """

    def __init__(self, dim: int, scan_type: str = "horizontal"):
        super().__init__()
        self.scan_type = scan_type
        self.proj_in = nn.Linear(dim, dim)
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=False
        )
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        self.proj_out = nn.Linear(dim, dim)

    def _apply_scan_direction(self, x: torch.Tensor) -> torch.Tensor:
        """Apply directional transformation before/after conv."""
        if self.scan_type == "horizontal":
            return x
        elif self.scan_type == "vertical":
            return x.transpose(2, 3)
        elif self.scan_type == "diagonal_lr":
            return x.flip(3)
        elif self.scan_type == "diagonal_rl":
            return x.flip(2)
        return x

    def _undo_scan_direction(self, x: torch.Tensor) -> torch.Tensor:
        """Undo directional transformation."""
        if self.scan_type == "horizontal":
            return x
        elif self.scan_type == "vertical":
            return x.transpose(2, 3)
        elif self.scan_type == "diagonal_lr":
            return x.flip(3)
        elif self.scan_type == "diagonal_rl":
            return x.flip(2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        x = self.proj_in(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self._apply_scan_direction(x)
        x = self.act(self.norm(self.dw_conv(x)))
        x = self._undo_scan_direction(x)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.proj_out(x)
        return x


# =============================================================================
# SwitchVSSBlock — Mixture-of-Scans
# =============================================================================


class SwitchVSSBlock(nn.Module):
    """
    SwitchVSSBlock: Mixture-of-Scans core innovation.

    4 scan experts with different scanning policies:
    horizontal, vertical, diagonal-LR, diagonal-RL.

    Lightweight MLP router: pooled features → softmax weights.
    Sparse activation: top-2 scan heads per token.
    Output: weighted sum of active scan heads.
    """

    def __init__(self, dim: int, drop_path: float = 0.0, top_k: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.top_k = top_k

        # 4 scan experts
        scan_types = ["horizontal", "vertical", "diagonal_lr", "diagonal_rl"]
        self.experts = nn.ModuleList(
            [ScanExpert(dim, scan_type=st) for st in scan_types]
        )
        self.num_experts = len(self.experts)

        # Router: lightweight MLP
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, self.num_experts),
        )

        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H, W, C)
        """
        residual = x
        x = self.norm(x)
        B, H, W, C = x.shape

        # Router weights
        x_pool = x.permute(0, 3, 1, 2).contiguous()
        router_logits = self.router(x_pool)  # (B, num_experts)
        router_weights = torch.softmax(router_logits, dim=-1)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(
            router_weights, self.top_k, dim=-1
        )
        # Renormalize top-k weights
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute all expert outputs (efficient for small number of experts)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)  # (B, E, H, W, C)

        # Gather top-k expert outputs and weight them
        out = torch.zeros(B, H, W, C, device=x.device, dtype=x.dtype)
        for k_idx in range(self.top_k):
            indices = topk_indices[:, k_idx]  # (B,)
            weights = topk_weights[:, k_idx]  # (B,)

            # Gather expert output for each batch element
            batch_indices = torch.arange(B, device=x.device)
            selected = expert_stack[batch_indices, indices]  # (B, H, W, C)
            out = out + selected * weights.view(B, 1, 1, 1)

        return residual + self.drop_path(out)


# =============================================================================
# CNN Branch
# =============================================================================


class CNNBranch(nn.Module):
    """CNN branch alongside SSM for local feature extraction."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, C, H, W)"""
        return self.conv(x)


# =============================================================================
# Encoder / Decoder Blocks
# =============================================================================


class EncoderBlock(nn.Module):
    """Encoder block: SwitchVSSBlock + CNN branch (parallel)."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.ssm = SwitchVSSBlock(dim, drop_path=drop_path)
        self.cnn = CNNBranch(dim)
        self.fuse = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, H, W, C)"""
        # SSM branch
        ssm_out = self.ssm(x)

        # CNN branch
        x_conv = x.permute(0, 3, 1, 2).contiguous()
        cnn_out = self.cnn(x_conv).permute(0, 2, 3, 1).contiguous()

        # Fuse
        fused = torch.cat([ssm_out, cnn_out], dim=-1)
        return self.fuse(fused)


class PatchMerging(nn.Module):
    """Downsample: 2x spatial reduction, 2x channel increase."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, H/2, W/2, 2C)"""
        B, H, W, C = x.shape
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


class PatchEmbed(nn.Module):
    """Patch embedding: Conv stem."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x


# =============================================================================
# Switch-UMamba Full Model
# =============================================================================


class SwitchUMamba(nn.Module):
    """
    Switch-UMamba: Dynamic Scanning Vision Mamba UNet.

    UNet architecture with SwitchVSSBlocks + CNN branches.
    Encoder: 4 stages + PatchMerging.
    Decoder: 4 stages + bilinear upsampling.
    Skip connections: concatenation (UNet style).
    Dropout2d(0.2) before final head.
    All weights random init — no pretrained.

    Output: raw logits (B, 1, H, W) — sigmoid NEVER inside forward().
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_dims: List[int] = None,
        encoder_depths: List[int] = None,
        dropout: float = 0.2,
        input_size: int = 256,
    ):
        super().__init__()
        if encoder_dims is None:
            encoder_dims = [96, 192, 384, 768]
        if encoder_depths is None:
            encoder_depths = [2, 2, 6, 2]

        self.encoder_dims = encoder_dims
        self.num_stages = len(encoder_dims)
        decoder_dims = list(reversed(encoder_dims))  # [768, 384, 192, 96]

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, encoder_dims[0], patch_size=4)

        # Encoder
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
                    EncoderBlock(encoder_dims[i], drop_path=dpr[cur + j])
                    for j in range(encoder_depths[i])
                ]
            )
            self.encoder_stages.append(blocks)
            cur += encoder_depths[i]

            if i < self.num_stages - 1:
                self.downsamples.append(PatchMerging(encoder_dims[i]))

        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(self.num_stages - 1):
            # Decoder stage operates on decoder_dims[i]
            dec_dim = decoder_dims[i]
            skip_dim = encoder_dims[self.num_stages - 2 - i]
            next_dim = decoder_dims[i + 1]

            # Upsample + channel projection
            self.upsample_layers.append(
                nn.Linear(dec_dim, next_dim)
            )

            # Skip concatenation: next_dim + skip_dim → next_dim
            self.skip_convs.append(
                nn.Sequential(
                    nn.Linear(next_dim + skip_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.GELU(),
                )
            )

            # Decoder blocks
            dec_blocks = nn.ModuleList(
                [EncoderBlock(next_dim) for _ in range(2)]
            )
            self.decoder_stages.append(dec_blocks)

        # Final head
        self.dropout2d = nn.Dropout2d(dropout)
        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_dims[0], encoder_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_dims[0]),
            nn.GELU(),
        )
        self.head = nn.Conv2d(encoder_dims[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights from scratch."""
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
            encoder_features.append(x)
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)

        # Decoder
        for i in range(self.num_stages - 1):
            B_dec, H_dec, W_dec, C_dec = x.shape

            # Upsample
            x = self.upsample_layers[i](x)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = x.permute(0, 2, 3, 1).contiguous()

            # Skip connection (concatenation — UNet style)
            skip = encoder_features[self.num_stages - 2 - i]

            # Handle size mismatch
            if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
                x_conv = x.permute(0, 3, 1, 2).contiguous()
                x_conv = F.interpolate(
                    x_conv,
                    size=(skip.shape[1], skip.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                )
                x = x_conv.permute(0, 2, 3, 1).contiguous()

            x = torch.cat([x, skip], dim=-1)
            x = self.skip_convs[i](x)

            # Decoder blocks
            for block in self.decoder_stages[i]:
                x = block(x)

        # Final upsampling to original resolution
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dropout2d(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        x = self.head(x)

        return x  # Raw logits — sigmoid NEVER inside forward()
