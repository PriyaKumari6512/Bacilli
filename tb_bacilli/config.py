"""
config.py — Single configuration dataclass for TB Bacilli Segmentation.

All hyperparameters in one place. Override via command-line or by editing defaults.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class TBConfig:
    """Complete configuration for TB bacilli segmentation pipeline."""

    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "cavmamba"  # "cavmamba" or "switch_umamba"
    # CaVMamba / VMamba-Small encoder dimensions
    encoder_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 15, 2])
    decoder_dims: List[int] = field(default_factory=lambda: [384, 192, 96, 48])
    decoder_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.2
    vmamba_pretrained_path: str = "./pretrained_weights/vmamba/vmamba_small_e238_ema.pth"
    num_classes: int = 1  # binary segmentation

    # ── Input ────────────────────────────────────────────────────────────
    input_size: int = 256  # 400→256 resize (validated, not 512)
    in_channels: int = 3

    # ── Data ─────────────────────────────────────────────────────────────
    data_root: str = "./data/DDS3"
    train_dir: str = "./data/DDS3/train"
    val_dir: str = "./data/DDS3/val"
    test_dir: str = "./data/DDS3/test"
    processed_dir: str = "./data/DDS3_processed"
    manifest_dir: str = "./data/DDS3_processed/manifests"
    pos_sample_weight: float = 3.0
    neg_sample_weight: float = 1.0
    stain_norm_method: str = "macenko"
    stain_reference_path: str = ""  # set to a training image path
    num_workers: int = 4
    multi_scale_factors: List[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25]
    )

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 150
    batch_size: int = 8
    encoder_lr: float = 6e-5
    decoder_lr: float = 6e-4
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 15
    early_stopping_monitor: str = "val_f1"
    cosine_T_0: int = 20
    cosine_T_mult: int = 2
    log_interval: int = 10  # batches
    overlay_interval: int = 5  # epochs
    seed: int = 42

    # ── Loss ─────────────────────────────────────────────────────────────
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    precision_penalty_weight: float = 0.2
    focal_gamma: float = 3.0
    focal_alpha: float = 0.9
    dice_smooth: float = 1.0
    precision_smooth: float = 1.0

    # ── Inference ────────────────────────────────────────────────────────
    confidence_threshold: float = 0.65
    tta_enabled: bool = True
    stain_normalizer_path: str = "./artifacts/stain_normalizer.pkl"
    sliding_window_size: int = 256
    sliding_window_stride: int = 128

    # ── Instance Extraction ──────────────────────────────────────────────
    min_instance_area: int = 20
    max_instance_area: int = 2000
    eccentricity_threshold: float = 0.7
    watershed_compactness: float = 0.001
    watershed_min_distance: int = 5
    morphological_kernel_size: int = 3

    # ── Paths ────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"
    artifact_dir: str = "./artifacts"

    def __post_init__(self):
        """Create necessary directories."""
        for d in [
            self.checkpoint_dir,
            self.log_dir,
            self.output_dir,
            self.artifact_dir,
            self.manifest_dir,
        ]:
            os.makedirs(d, exist_ok=True)
