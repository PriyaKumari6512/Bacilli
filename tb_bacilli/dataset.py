"""
dataset.py — Dataset and DataLoader for TB Bacilli Segmentation.

- Loads from JSON manifests
- Returns image tensor, mask tensor float32 {0.0, 1.0}, metadata
- Assert mask values are binary in __getitem__
- WeightedRandomSampler by bacilli_ratio — train split only
- Multi-scale training: randomly pick scale, resize, back to 256
"""

import json
import logging
import os
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class TBBacilliDataset(Dataset):
    """
    TB Bacilli segmentation dataset.

    Loads from JSON manifest with image_path, mask_path, bacilli_ratio.
    Returns image tensor (C, H, W), mask tensor (1, H, W) float32 {0, 1}, metadata.
    """

    def __init__(
        self,
        manifest_path: str,
        transforms=None,
        input_size: int = 256,
        multi_scale: bool = False,
        scale_factors: list = None,
    ):
        """
        Args:
            manifest_path: path to JSON manifest file.
            transforms: albumentations Compose.
            input_size: target size (256).
            multi_scale: enable multi-scale training.
            scale_factors: list of scale factors [0.75, 1.0, 1.25].
        """
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.transforms = transforms
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.scale_factors = scale_factors or [0.75, 1.0, 1.25]

        logger.info(
            f"Dataset loaded: {len(self.manifest)} samples from {manifest_path}"
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        entry = self.manifest[idx]
        img_path = entry["image_path"]
        mask_path = entry["mask_path"]

        # Read image (BGR → RGB)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Binarize mask: {0, 255} → {0, 1}
        mask = (mask > 127).astype(np.uint8)

        # Multi-scale training
        if self.multi_scale and self.transforms is not None:
            scale = random.choice(self.scale_factors)
            if scale != 1.0:
                scaled_size = int(self.input_size * scale)
                img = cv2.resize(
                    img, (scaled_size, scaled_size), interpolation=cv2.INTER_LINEAR
                )
                mask = cv2.resize(
                    mask, (scaled_size, scaled_size), interpolation=cv2.INTER_NEAREST
                )
                # Resize back to input_size
                img = cv2.resize(
                    img,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = cv2.resize(
                    mask,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_NEAREST,
                )

        # Apply augmentations
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img_tensor = transformed["image"]  # (C, H, W) float32
            mask_np = transformed["mask"]  # (H, W) uint8 or float
        else:
            # Manual conversion
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_np = mask

        # Ensure mask is float32 {0.0, 1.0}
        if isinstance(mask_np, torch.Tensor):
            mask_tensor = mask_np.float().unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)

        # ASSERT: mask values must be binary {0, 1}
        unique_vals = torch.unique(mask_tensor)
        assert all(
            v in [0.0, 1.0] for v in unique_vals.tolist()
        ), f"Mask values must be {{0.0, 1.0}}, got {unique_vals.tolist()}"

        metadata = {
            "image_path": img_path,
            "mask_path": mask_path,
            "bacilli_ratio": entry.get("bacilli_ratio", 0.0),
        }

        return img_tensor, mask_tensor, metadata


def get_weighted_sampler(manifest_path: str, config) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler based on bacilli_ratio.

    Positive images get pos_sample_weight, negatives get neg_sample_weight.
    Train split only.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    weights = []
    for entry in manifest:
        if entry["bacilli_ratio"] > 0:
            weights.append(config.pos_sample_weight)
        else:
            weights.append(config.neg_sample_weight)

    weights = torch.FloatTensor(weights)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def get_dataloaders(config, train_transforms=None, val_transforms=None):
    """
    Create train, val, test dataloaders.

    Train uses WeightedRandomSampler + multi-scale.
    Val/Test use sequential loading.
    """
    train_manifest = os.path.join(config.manifest_dir, "train.json")
    val_manifest = os.path.join(config.manifest_dir, "val.json")
    test_manifest = os.path.join(config.manifest_dir, "test.json")

    dataloaders = {}

    if os.path.isfile(train_manifest):
        train_dataset = TBBacilliDataset(
            train_manifest,
            transforms=train_transforms,
            input_size=config.input_size,
            multi_scale=True,
            scale_factors=config.multi_scale_factors,
        )
        train_sampler = get_weighted_sampler(train_manifest, config)
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,  # WeightedRandomSampler on train only
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if os.path.isfile(val_manifest):
        val_dataset = TBBacilliDataset(
            val_manifest,
            transforms=val_transforms,
            input_size=config.input_size,
            multi_scale=False,
        )
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    if os.path.isfile(test_manifest):
        test_dataset = TBBacilliDataset(
            test_manifest,
            transforms=val_transforms,
            input_size=config.input_size,
            multi_scale=False,
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return dataloaders
