"""
augmentations.py — Data augmentation pipelines for TB Bacilli Segmentation.

TRAIN: geometric + ZN stain variation + microscope noise.
VAL/TEST: resize + normalize only.
TTA: 8-fold (flips + rotations), averaged sigmoid probabilities.

NO ElasticTransform (bacilli are rigid rods).
NO heavy color jitter (ZN red-blue is diagnostically meaningful).
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

logger = logging.getLogger(__name__)

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(input_size: int = 256) -> "A.Compose":
    """
    Training augmentation pipeline.

    - Geometric: flips, rotations, shift-scale-rotate
    - ZN stain variation: HSV, brightness-contrast, CLAHE
    - Microscope noise: Gauss noise, Gauss blur, ISO noise
    - Normalize (ImageNet) + ToTensorV2
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentations.")

    return A.Compose(
        [
            A.Resize(input_size, input_size),
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=4,  # cv2.BORDER_REFLECT
                p=0.5,
            ),
            # ZN stain variation
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            # Microscope noise
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.15), p=0.2),
            # Normalize and convert
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(input_size: int = 256) -> "A.Compose":
    """Validation/Test augmentation: resize + normalize only."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentations.")

    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# =============================================================================
# Test-Time Augmentation
# =============================================================================


class TTAWrapper:
    """
    Test-Time Augmentation wrapper.

    8 augmentations: 4 rotations × 2 flips.
    Averages sigmoid probabilities across all 8 before thresholding.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    def _augment(
        self, x: torch.Tensor, flip: bool, rot: int
    ) -> torch.Tensor:
        """Apply augmentation: optional horizontal flip + k*90° rotation."""
        if flip:
            x = torch.flip(x, dims=[3])  # horizontal flip
        if rot > 0:
            x = torch.rot90(x, k=rot, dims=[2, 3])
        return x

    def _deaugment(
        self, x: torch.Tensor, flip: bool, rot: int
    ) -> torch.Tensor:
        """Reverse augmentation."""
        if rot > 0:
            x = torch.rot90(x, k=4 - rot, dims=[2, 3])
        if flip:
            x = torch.flip(x, dims=[3])
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        TTA prediction: average sigmoid probabilities over 8 augmentations.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, 1, H, W) averaged probability map.
        """
        self.model.eval()
        x = x.to(self.device)
        prob_sum = torch.zeros(
            x.shape[0], 1, x.shape[2], x.shape[3],
            device=self.device, dtype=x.dtype,
        )
        count = 0

        for flip in [False, True]:
            for rot in [0, 1, 2, 3]:
                x_aug = self._augment(x, flip, rot)
                logits = self.model(x_aug)
                probs = torch.sigmoid(logits)
                probs = self._deaugment(probs, flip, rot)
                prob_sum += probs
                count += 1

        return prob_sum / count
