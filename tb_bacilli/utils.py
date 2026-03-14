"""
utils.py — Utility functions for TB Bacilli Segmentation pipeline.

Includes: seed, metrics, checkpointing, Macenko fallback, early stopping,
and training curve plotting.
"""

import os
import random
import pickle
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# =============================================================================
# AverageMeter
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved: {filepath}")
    if is_best and best_path:
        torch.save(state, best_path)
        logger.info(f"Best checkpoint saved: {best_path}")


def load_checkpoint(filepath: str, model, optimizer=None, scheduler=None):
    """Load training checkpoint. Returns epoch and best metric."""
    if not os.path.isfile(filepath):
        logger.warning(f"No checkpoint found at {filepath}")
        return 0, 0.0

    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    logger.info(f"Loaded checkpoint from epoch {epoch}, best_metric={best_metric:.4f}")
    return epoch, best_metric


# =============================================================================
# Pixel Metrics
# =============================================================================

def compute_pixel_metrics(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-level segmentation metrics.

    Args:
        pred: predicted probability map (H, W), values in [0, 1].
        target: ground truth binary mask (H, W), values in {0, 1}.
        threshold: binarization threshold.

    Returns:
        Dictionary with dice, iou, precision, recall, f1.
    """
    pred_bin = (pred >= threshold).astype(np.float32)
    target = target.astype(np.float32)

    tp = np.sum(pred_bin * target)
    fp = np.sum(pred_bin * (1 - target))
    fn = np.sum((1 - pred_bin) * target)

    smooth = 1e-7
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    dice = 2 * tp / (2 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_fp_rate_on_negatives(
    preds: list, targets: list, threshold: float = 0.5
) -> float:
    """
    Compute false positive rate on negative (all-background) images.

    Among images where GT is all zero, what fraction has any predicted positive?

    Args:
        preds: list of predicted probability maps (H, W).
        targets: list of ground truth binary masks (H, W).
        threshold: binarization threshold.

    Returns:
        FP rate on negatives (0.0 to 1.0).
    """
    neg_count = 0
    fp_on_neg = 0

    for pred, target in zip(preds, targets):
        if np.sum(target) == 0:  # negative image
            neg_count += 1
            pred_bin = (pred >= threshold).astype(np.float32)
            if np.sum(pred_bin) > 0:
                fp_on_neg += 1

    if neg_count == 0:
        return 0.0
    return fp_on_neg / neg_count


# =============================================================================
# Pure Numpy Macenko Stain Normalization (fallback)
# =============================================================================

class MacenkoNormalizerNumpy:
    """
    Pure NumPy Macenko stain normalization.

    SVD-based optical density normalization — no external dependency.
    Reference: Macenko et al., ISBI 2009.
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.maxC_target = None

    def _rgb_to_od(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        img = img.astype(np.float64)
        img = np.clip(img, 1, 255)
        return -np.log(img / 255.0)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density back to RGB."""
        rgb = np.exp(-od) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _get_stain_matrix(self, img: np.ndarray, beta: float = 0.15, alpha: float = 1.0):
        """Extract stain matrix using SVD on optical density."""
        od = self._rgb_to_od(img)
        od_flat = od.reshape(-1, 3)

        # Remove pixels with low OD (background)
        od_thresh = od_flat[np.all(od_flat > beta, axis=1)]
        if len(od_thresh) < 10:
            return np.eye(3)[:2], np.ones(2)

        # SVD
        _, _, Vt = np.linalg.svd(od_thresh, full_matrices=False)
        V = Vt[:2, :]  # top 2 eigenvectors

        # Project
        proj = od_thresh @ V.T
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)

        v1 = V.T @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = V.T @ np.array([np.cos(max_phi), np.sin(max_phi)])

        # Ensure H&E order
        if v1[0] > v2[0]:
            stain_matrix = np.array([v1, v2])
        else:
            stain_matrix = np.array([v2, v1])

        # Concentrations
        C = np.linalg.lstsq(stain_matrix.T, od_flat.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)

        return stain_matrix, maxC

    def fit(self, target_img: np.ndarray):
        """Fit normalizer to a reference image."""
        self.stain_matrix_target, self.maxC_target = self._get_stain_matrix(
            target_img
        )

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to reference stain."""
        if self.stain_matrix_target is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        stain_matrix_src, maxC_src = self._get_stain_matrix(img)
        od = self._rgb_to_od(img)
        od_flat = od.reshape(-1, 3)

        # Source concentrations
        C = np.linalg.lstsq(stain_matrix_src.T, od_flat.T, rcond=None)[0]

        # Normalize concentrations
        maxC_src = np.clip(maxC_src, 1e-6, None)
        C = C / maxC_src[:, None] * self.maxC_target[:, None]

        # Reconstruct
        od_norm = self.stain_matrix_target.T @ C
        rgb_norm = self._od_to_rgb(od_norm.T.reshape(img.shape))
        return rgb_norm

    def save(self, filepath: str):
        """Save fitted normalizer to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "stain_matrix_target": self.stain_matrix_target,
                    "maxC_target": self.maxC_target,
                },
                f,
            )
        logger.info(f"Stain normalizer saved to {filepath}")

    def load(self, filepath: str):
        """Load fitted normalizer from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.stain_matrix_target = data["stain_matrix_target"]
        self.maxC_target = data["maxC_target"]
        logger.info(f"Stain normalizer loaded from {filepath}")


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping based on a monitored metric.

    Args:
        patience: epochs to wait after last improvement.
        monitor: metric name.
        mode: 'max' or 'min'.
    """

    def __init__(self, patience: int = 15, monitor: str = "val_f1", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update with new metric value. Returns True if should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered: {self.monitor} did not improve "
                    f"for {self.patience} epochs. Best: {self.best_value:.4f}"
                )
                return True
        return False


# =============================================================================
# Training Curve Plotting
# =============================================================================

def plot_training_curves(log_dict: Dict[str, list], save_path: str):
    """Plot and save training curves from log dictionary."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    if "train_loss" in log_dict:
        axes[0, 0].plot(log_dict["train_loss"], label="Train Loss")
    if "val_loss" in log_dict:
        axes[0, 0].plot(log_dict["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Dice & IoU
    if "val_dice" in log_dict:
        axes[0, 1].plot(log_dict["val_dice"], label="Val Dice")
    if "val_iou" in log_dict:
        axes[0, 1].plot(log_dict["val_iou"], label="Val IoU")
    axes[0, 1].set_title("Dice & IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Precision & Recall
    if "val_precision" in log_dict:
        axes[0, 2].plot(log_dict["val_precision"], label="Val Precision")
    if "val_recall" in log_dict:
        axes[0, 2].plot(log_dict["val_recall"], label="Val Recall")
    if "val_f1" in log_dict:
        axes[0, 2].plot(log_dict["val_f1"], label="Val F1")
    axes[0, 2].set_title("Precision / Recall / F1")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # FP rate on negatives
    if "val_fp_rate_on_negatives" in log_dict:
        axes[1, 0].plot(
            log_dict["val_fp_rate_on_negatives"], label="FP Rate on Negatives", color="red"
        )
        axes[1, 0].axhline(y=0.01, color="green", linestyle="--", label="1% Target")
    axes[1, 0].set_title("FP Rate on Negatives")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Loss components
    for key in ["train_dice_loss", "train_focal_loss", "train_precision_penalty"]:
        if key in log_dict:
            axes[1, 1].plot(log_dict[key], label=key)
    axes[1, 1].set_title("Loss Components")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Learning rates
    if "encoder_lr" in log_dict:
        axes[1, 2].plot(log_dict["encoder_lr"], label="Encoder LR")
    if "decoder_lr" in log_dict:
        axes[1, 2].plot(log_dict["decoder_lr"], label="Decoder LR")
    axes[1, 2].set_title("Learning Rates")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    axes[1, 2].set_yscale("log")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {save_path}")
