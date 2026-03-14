"""
loss.py — TBSegLoss: Combined loss for TB bacilli segmentation.

Three terms:
- Dice Loss (weight=0.5): handles class imbalance
- Focal Loss (weight=0.3): penalizes false positives (gamma=3.0)
- Precision Penalty (weight=0.2): directly penalizes FP on negative images

Also includes BinaryDice metric class for monitoring.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Soft Dice Loss on sigmoid(pred)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: raw logits (B, 1, H, W).
            target: binary mask (B, 1, H, W), values in {0.0, 1.0}.
        """
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    gamma=3.0: stronger penalization of false positives.
    alpha=0.9: weight toward positive class.
    """

    def __init__(self, gamma: float = 3.0, alpha: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: raw logits (B, 1, H, W).
            target: binary mask (B, 1, H, W), values in {0.0, 1.0}.
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_sig = torch.sigmoid(pred)
        pt = target * pred_sig + (1 - target) * (1 - pred_sig)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class PrecisionPenalty(nn.Module):
    """
    Precision penalty: 1 - TP / (TP + FP + smooth).

    Directly penalizes false positives.
    Critical fix for predictions on negative images.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: raw logits (B, 1, H, W).
            target: binary mask (B, 1, H, W), values in {0.0, 1.0}.
        """
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        precision = tp / (tp + fp + self.smooth)
        return 1.0 - precision


class TBSegLoss(nn.Module):
    """
    Combined TB Segmentation Loss.

    Total = dice_weight * DiceLoss
          + focal_weight * FocalLoss
          + precision_penalty_weight * PrecisionPenalty

    Logs each term separately. Warns if precision_penalty stays high.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        precision_penalty_weight: float = 0.2,
        focal_gamma: float = 3.0,
        focal_alpha: float = 0.9,
        dice_smooth: float = 1.0,
        precision_smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.precision_penalty_weight = precision_penalty_weight

        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.precision_penalty = PrecisionPenalty(smooth=precision_smooth)

        # Track precision penalty for warnings
        self._pp_high_count = 0

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: raw logits (B, 1, H, W).
            target: binary mask (B, 1, H, W), values in {0.0, 1.0}.

        Returns:
            Dict with 'total', 'dice_loss', 'focal_loss', 'precision_penalty'.
        """
        d_loss = self.dice_loss(pred, target)
        f_loss = self.focal_loss(pred, target)
        pp_loss = self.precision_penalty(pred, target)

        total = (
            self.dice_weight * d_loss
            + self.focal_weight * f_loss
            + self.precision_penalty_weight * pp_loss
        )

        # Track high precision penalty
        pp_val = pp_loss.item()
        if pp_val > 0.3:
            self._pp_high_count += 1
            if self._pp_high_count >= 5:
                logger.warning(
                    f"Precision penalty > 0.3 for {self._pp_high_count} "
                    f"consecutive steps. Check for excessive false positives."
                )
        else:
            self._pp_high_count = 0

        return {
            "total": total,
            "dice_loss": d_loss.detach(),
            "focal_loss": f_loss.detach(),
            "precision_penalty": pp_loss.detach(),
        }


class BinaryDice(nn.Module):
    """Binary Dice metric for monitoring (not a loss)."""

    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    @torch.no_grad()
    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
    ) -> float:
        """
        Args:
            pred: probability map after sigmoid (B, 1, H, W).
            target: binary mask (B, 1, H, W).
            threshold: binarization threshold.

        Returns:
            Dice score as float.
        """
        pred_bin = (pred >= threshold).float()
        pred_flat = pred_bin.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return dice.item()
