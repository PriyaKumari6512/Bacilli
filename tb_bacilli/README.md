# TB Bacilli Segmentation Pipeline

**Production-ready two-stage pipeline** for TB bacilli segmentation, instance extraction, bounding box detection, confidence scoring, and counting on the DDS3 dataset.

## Architecture

### Two-Stage Pipeline

- **Stage 1**: Binary segmentation model trained on DDS3 binary masks → probability map per pixel
- **Stage 2**: Instance extraction from probability map (no training) → watershed + connected components → per-instance outputs

### Models

1. **CaVMamba** (Primary) — CNN-augmented VMamba with SandwichBlocks and Dynamic Feature Fusion. Pretrained from VMamba-Small encoder weights.
2. **Switch-UMamba** (Fallback) — Dynamic scanning UNet with Mixture-of-Scans routing. Trains from scratch, no pretrained weights needed.

## Quick Start

### 1. Environment Setup

```bash
bash setup_env.sh
conda activate tb_bacilli
```

### 2. Download Pretrained Weights (CaVMamba only)

```bash
bash download_weights.sh
```

### 3. Prepare Data

Place DDS3 dataset in `./data/DDS3/` with `train/`, `val/`, `test/` splits, each containing `images/` and `masks/` subdirectories.

```bash
python prepare_data.py
```

### 4. Train

```bash
python train.py
# or with Switch-UMamba:
python train.py --model switch_umamba
```

### 5. Evaluate

```bash
python evaluate.py --tta --save_vis
```

### 6. Inference

```bash
# Single image
python inference.py --input path/to/image.png --output ./results

# Batch
python inference.py --input path/to/image_dir/ --output ./results
```

## Output Per Image

| Output | Description |
|--------|-------------|
| Bounding box | `(x1, y1, x2, y2)` per bacillus instance |
| Pixel mask | Binary mask per instance |
| Confidence | Mean probability from Stage 1 map within instance region |
| Count | Number of valid instances passing all filters |

## DDS3 Dataset Structure

```
data/DDS3/
├── train/         (6000 images)
│   ├── images/
│   └── masks/
├── val/           (1000 images)
│   ├── images/
│   └── masks/
└── test/          (1000 images)
    ├── images/
    └── masks/
```

- 400×400 mosaic images (10×10 grid of 40×40 sub-patches)
- Binary masks: bacilli = white, background = black
- ZN stained: red-pink bacilli on blue background
- Severe class imbalance: bacilli pixels < 2-5%

## Key Design Decisions

1. **INTER_NEAREST** for all mask resizing — never blur binary masks
2. **Sigmoid outside forward()** — models output raw logits
3. **Default threshold = 0.65** — tuned for ZN-stained bacilli
4. **Train on 400→256 resize** — never extract 40×40 tiles
5. **WeightedRandomSampler** on train only — oversample positive images (3:1)
6. **Differential LR**: encoder 6e-5, decoder 6e-4
7. **val_fp_rate_on_negatives** logged every epoch — critical FP monitoring
8. **Macenko stain normalization** in both prepare_data.py and inference.py
9. **Precision penalty loss** — directly penalizes false positives
10. **Instance confidence** = mean prob_map within instance pixels

## Training Details

- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=20, T_mult=2)
- **Loss**: Dice (0.5) + Focal (0.3, γ=3.0, α=0.9) + Precision Penalty (0.2)
- **Early Stopping**: patience=15 on val_f1
- **Mixed Precision**: enabled by default
- **Augmentations**: geometric + ZN stain variation + microscope noise
- **TTA**: 8-fold (4 rotations × 2 flips), averaged sigmoid probabilities

## Instance Extraction Pipeline

1. Threshold probability map at 0.65
2. Morphological closing (fill rod gaps)
3. Distance transform
4. Local maxima as watershed seeds
5. Watershed segmentation (splits touching bacilli)
6. Per-region: bbox, mask, area, eccentricity, confidence
7. Filter: area 20–2000px, eccentricity ≥ 0.7
8. Sort by confidence descending

## Project Structure

```
tb_bacilli/
├── setup_env.sh              # Environment setup
├── download_weights.sh       # Pretrained weight download
├── config.py                 # Configuration dataclass
├── prepare_data.py           # Data preparation + stain normalization
├── dataset.py                # Dataset + weighted sampling
├── augmentations.py          # Augmentation pipelines + TTA
├── models/
│   ├── __init__.py
│   ├── cavmamba.py           # CaVMamba (primary)
│   ├── switch_umamba.py      # Switch-UMamba (fallback)
│   └── model_factory.py      # Model factory with auto-fallback
├── loss.py                   # TBSegLoss (dice + focal + precision penalty)
├── train.py                  # Training loop
├── evaluate.py               # Evaluation + visualizations
├── inference.py              # Single/batch inference
├── instance_extraction.py    # Watershed-based instance extraction
├── postprocess.py            # Post-processing + visualization
├── utils.py                  # Utilities (metrics, Macenko, checkpointing)
└── README.md                 # This file
```
