```
You are an expert computer vision and medical image analysis engineer.
Generate a complete production-ready Python codebase for TB bacilli
segmentation, instance extraction, bounding box detection, confidence
scoring and counting on the DDS3 dataset.

═══════════════════════════════════════════════════════════════
CRITICAL CONTEXT — READ BEFORE WRITING ANYTHING
═══════════════════════════════════════════════════════════════

DDS3 DATASET FACTS:
- 400×400 mosaic images arranged as 10×10 grid of 40×40 sub-patches
- Each image has ONE binary mask: bacilli=white, background=black
- DDS3 is SEMANTIC segmentation — no instance labels, no bboxes,
  no per-bacillus annotations exist in DDS3
- Pre-defined splits: 6000 train / 1000 val / 1000 test — use exactly
- ZN stained: red-pink bacilli on blue background
- Severe class imbalance: bacilli pixels <2-5% of total image
- TRAIN ON FULL 400×400 images resized to 256×256
- NEVER extract 40×40 tiles and upscale — pure interpolation artifacts

TWO-STAGE PIPELINE (mandatory given DDS3 structure):
Stage 1 — Binary segmentation model trained on DDS3 binary masks
          outputs probability map per pixel
Stage 2 — Instance extraction from probability map (no training)
          watershed + connected components → per-instance outputs

REQUIRED FINAL OUTPUT PER IMAGE:
- Bounding box per bacillus instance
- Pixel mask per bacillus instance
- Confidence score per bacillus instance
  (mean probability from Stage 1 map within that instance region)
- Count = valid instances above confidence threshold
  AND passing shape filters

PREVIOUS TRAINING FAILURE:
Model predicted on negative (all-background) images.
Fix at: architecture level, loss level, sampling level,
threshold level, and post-processing level.

═══════════════════════════════════════════════════════════════
PROJECT STRUCTURE
═══════════════════════════════════════════════════════════════

tb_bacilli/
├── setup_env.sh
├── download_weights.sh
├── config.py
├── prepare_data.py
├── dataset.py
├── augmentations.py
├── models/
│   ├── __init__.py
│   ├── cavmamba.py
│   ├── switch_umamba.py
│   └── model_factory.py
├── loss.py
├── train.py
├── evaluate.py
├── inference.py
├── instance_extraction.py
├── postprocess.py
├── utils.py
└── README.md

═══════════════════════════════════════════════════════════════
setup_env.sh
═══════════════════════════════════════════════════════════════

Create conda environment with all dependencies.
Install in correct order — causal-conv1d before mamba-ssm.
Note: mamba-ssm requires Linux + CUDA. Add WSL2 note for Windows users.

═══════════════════════════════════════════════════════════════
download_weights.sh
═══════════════════════════════════════════════════════════════

Download VMamba-Small pretrained weights from:
https://github.com/MzeroMiko/VMamba (official repo, GitHub releases)
Save to: ./pretrained_weights/vmamba/

If VMamba download fails, fallback to MambaVision from HuggingFace:
nvidia/MambaVision-T-1K via transformers AutoModel
Save encoder weights only to: ./pretrained_weights/mambavision/

Script must be idempotent — skip if weights already exist.

═══════════════════════════════════════════════════════════════
config.py
═══════════════════════════════════════════════════════════════

Single config dataclass with ALL hyperparameters:

MODEL:
- model_name: "cavmamba" or "switch_umamba"
- cavmamba dims and depths matching VMamba-Small
- dropout: 0.2
- vmamba pretrained weight path

INPUT:
- input_size: 256  (400→256 is validated, not 512 — no benefit + OOM)
- binary output: 1 class

DATA:
- dds3 split directories
- pos_sample_weight: 3.0, neg_sample_weight: 1.0
- stain_norm_method: macenko
- stain reference image path

TRAINING:
- encoder_lr: 6e-5, decoder_lr: 6e-4
- early stopping patience: 15 on val F1
- mixed precision, gradient clipping

LOSS:
- dice_weight: 0.5, focal_weight: 0.3, precision_penalty_weight: 0.2
- focal_gamma: 3.0, focal_alpha: 0.9

INFERENCE:
- confidence_threshold: 0.65
- tta_enabled: True

INSTANCE EXTRACTION:
- min_instance_area: 20, max_instance_area: 2000
- eccentricity_threshold: 0.7
- watershed_compactness: 0.001

═══════════════════════════════════════════════════════════════
prepare_data.py
═══════════════════════════════════════════════════════════════

- Apply Macenko stain normalization to every image
- Fit normalizer on one reference training image
- Save fitted normalizer to ./artifacts/stain_normalizer.pkl
- If staintools unavailable → pure numpy Macenko fallback
  (SVD-based OD normalization, no external dependency)
- Resize images 400→256 INTER_LINEAR
- Resize masks 400→256 INTER_NEAREST (never smooth masks)
- Print class imbalance stats per split
- Save JSON manifest per split: image_path, mask_path,
  bacilli_pixel_count, bacilli_ratio
- Idempotent — skip if outputs exist

═══════════════════════════════════════════════════════════════
dataset.py
═══════════════════════════════════════════════════════════════

- Load from JSON manifests
- Return image tensor, mask tensor float32 {0.0, 1.0}, metadata
- Assert mask values are binary in __getitem__
- WeightedRandomSampler by bacilli_ratio — train split only
- Multi-scale training: randomly pick scale from [0.75, 1.0, 1.25],
  resize to scaled size then back to 256
  INTER_LINEAR for image, INTER_NEAREST for mask
  (teaches model to recognize bacilli at different apparent sizes —
  critical for generalization to larger clinical images)

═══════════════════════════════════════════════════════════════
augmentations.py
═══════════════════════════════════════════════════════════════

TRAIN pipeline (albumentations):
- Geometric: HorizontalFlip, VerticalFlip, RandomRotate90,
  ShiftScaleRotate with BORDER_REFLECT
- ZN stain variation: HueSaturationValue, RandomBrightnessContrast, CLAHE
- Microscope noise: GaussNoise, GaussianBlur, ISONoise
- Normalize ImageNet stats, ToTensorV2

DO NOT add:
- ElasticTransform (bacilli are rigid rods)
- Heavy color jitter (ZN red-blue is diagnostically meaningful)

VAL/TEST: Resize + Normalize + ToTensorV2 only

TTA: TTAWrapper class, 8 augmentations (flips + rotations),
average sigmoid probabilities across all 8 before thresholding

═══════════════════════════════════════════════════════════════
models/cavmamba.py  ← PRIMARY
═══════════════════════════════════════════════════════════════

Implement CaVMamba from:
"CaVMamba: Visual State Space Model with CNN Augmented VMamba"
The Visual Computer, 2025.

Components:

VSSBlock:
Standard VMamba 2D-Selective-Scan scanning in 4 directions

SandwichBlock (core innovation):
LayerNorm → DepthwiseConv3×3 → GELU → VSSBlock → DepthwiseConv3×3
→ GELU → residual add
First CNN: local rod-shape edges and stain features
VMamba: global context → learns empty region suppression
        (architectural fix for false positives on negative images)
Second CNN: refines local detail after global context injection

DynamicFeatureFusion:
Upsample all encoder stages to same size → concat → Conv1×1 →
Softmax weights → weighted sum
Learns which scale to trust per spatial location dynamically

Full model:
Encoder: 4 stages of SandwichBlocks + patch merging
Dims and depths matching VMamba-Small
Decoder: 4 stages of SandwichBlocks + patch expanding
Skip connections: additive
Final head: Conv2d → 1 channel + Dropout2d(0.2) before head

Pretrained loading:
Load VMamba-Small encoder weights, map keys to CaVMamba encoder
Log loaded vs skipped weights, use strict=False
Decoder initialized from scratch

forward(): input (B,3,256,256) → raw logits (B,1,256,256)
Sigmoid NEVER inside forward()

═══════════════════════════════════════════════════════════════
models/switch_umamba.py  ← FALLBACK
═══════════════════════════════════════════════════════════════

Implement Switch-UMamba from:
"Switch-UMamba: Dynamic Scanning Vision Mamba UNet"
ScienceDirect 2025.

IMPORTANT: Switch-UMamba achieves SOTA WITHOUT pretrained weights.
Train entirely from scratch — do not load any pretrained weights.

SwitchVSSBlock (core innovation — Mixture-of-Scans):
4 scan experts each with different scanning policy:
horizontal, vertical, diagonal-LR, diagonal-RL
Lightweight MLP router: takes pooled features → softmax weights
Sparse activation: top-2 scan heads per token
Output: weighted sum of active scan heads
This handles arbitrary-angle bacilli that static scanning misses

Full model:
UNet architecture with SwitchVSSBlocks + CNN branches alongside SSM
Encoder: 4 stages + patch merging downsampling
Decoder: 4 stages + bilinear upsampling
Skip connections: concatenation (UNet style)
Dropout2d(0.2) before final head
All weights random init

forward(): (B,3,256,256) → raw logits (B,1,256,256)
Sigmoid NEVER inside forward()

═══════════════════════════════════════════════════════════════
models/model_factory.py
═══════════════════════════════════════════════════════════════

get_model(config):
- cavmamba: load + attempt pretrained weight loading
  if weight file missing → warn, continue with random init
  if mamba-ssm ImportError → auto-switch to switch_umamba
- switch_umamba: random init, no pretrained
- Print param summary
- Return model

═══════════════════════════════════════════════════════════════
loss.py
═══════════════════════════════════════════════════════════════

TBSegLoss — three terms:

Dice Loss (0.5): smooth=1.0, on sigmoid(pred)

Focal Loss (0.3): gamma=3.0, alpha=0.9
Higher gamma = stronger false positive penalization

Precision Penalty (0.2):
1 - (TP / (TP + FP + smooth)) on sigmoid(pred)
Directly penalizes false positives
Critical fix for predictions on negative images

Log each term separately every epoch.
If precision_penalty > 0.3 for 5+ consecutive epochs → log warning.

BinaryDice metric class for monitoring (not a loss).

═══════════════════════════════════════════════════════════════
train.py
═══════════════════════════════════════════════════════════════

- AdamW differential LR: encoder=6e-5, decoder=6e-4
  (encoder has pretrained weights → gentle tuning
   decoder trains from scratch → faster learning)
- CosineAnnealingWarmRestarts
- Mixed precision + gradient clipping
- Early stopping patience=15 on val_f1

Log every epoch:
  train: loss + all 3 loss terms
  val: loss, dice, iou, f1, precision, recall
  val_fp_rate_on_negatives ← CRITICAL (previous failure metric)
  (among GT-all-zero images: fraction with any predicted positives)
  encoder_lr, decoder_lr

Checkpointing: best (val_f1) + last (every epoch)
Resume: auto-resume from last checkpoint if exists
TensorBoard: all metrics + sample overlays every 5 epochs

═══════════════════════════════════════════════════════════════
instance_extraction.py
═══════════════════════════════════════════════════════════════

extract_instances(prob_map, config) → list of instance dicts

Step 1: Threshold at config.confidence_threshold → binary mask
Step 2: Morphological closing (fill gaps along rod axis)
Step 3: Distance transform
Step 4: Local maxima as watershed seeds (min_distance=5px)
Step 5: Watershed segmentation → splits touching bacilli
Step 6: Label connected components
Step 7: Per region compute:
        bbox (x1,y1,x2,y2), mask, area, eccentricity,
        confidence = mean(prob_map[instance_pixels])
Step 8: Filter:
        area < min_instance_area → remove (noise)
        area > max_instance_area → remove (not bacilli)
        eccentricity < eccentricity_threshold → remove
        (bacilli are rods: high eccentricity, not circular)
Step 9: Sort by confidence descending

Return: list of {bbox, mask, confidence, area, eccentricity, instance_id}
count = len(list)

═══════════════════════════════════════════════════════════════
evaluate.py
═══════════════════════════════════════════════════════════════

Pixel metrics: Dice, IoU, Precision, Recall, F1
fp_rate_on_negatives: reported as standalone metric

Threshold sweep 0.3→0.95:
- All metrics at each threshold
- Optimal threshold by max F1
- Threshold where fp_rate_on_negatives < 1%
- Save curves as PNG

Instance metrics:
- Per-bacillus precision/recall (IoU>0.5 match)
- Mean absolute count error
- Count Pearson R correlation

Visualizations:
- Overlay: image + GT (green) + predicted instances (colored per instance)
  + confidence text on each bbox
- TP / FP / FN detection grids
- Negative image prediction grid (must be empty after fixes)

═══════════════════════════════════════════════════════════════
inference.py
═══════════════════════════════════════════════════════════════

MODE 1 — Small image (~400×400):
- Load stain normalizer from ./artifacts/stain_normalizer.pkl
- Macenko normalize
- Resize to 256×256
- Stage 1 model forward
- TTA if enabled
- Stage 2 instance_extraction()
- Return instances + count

MODE 2 — Large clinical image (>400×400):
- Macenko normalize first
- Sliding window: window=256, stride=128, reflect padding
- Overlap-averaged probability map
- Upsample back to original size
- Stage 2 instance_extraction()
- Return full-size mask + instances + count

Batch inference:
- Accept image directory
- Auto-detect mode by image size
- Output per image: mask PNG, overlay, JSON with instances and count

═══════════════════════════════════════════════════════════════
utils.py
═══════════════════════════════════════════════════════════════

- set_seed(42)
- AverageMeter
- save_checkpoint / load_checkpoint
- compute_pixel_metrics(pred, target)
- compute_fp_rate_on_negatives(pred, target)
- MacenkoNormalizerNumpy (pure numpy fallback):
  fit, transform, save, load
- EarlyStopping(patience, monitor='val_f1', mode='max')
- plot_training_curves(log_dict, save_path)

═══════════════════════════════════════════════════════════════
NON-NEGOTIABLE CONSTRAINTS
═══════════════════════════════════════════════════════════════

1. INTER_NEAREST for all mask resizing — never blur binary masks
2. Sigmoid outside forward() — never inside
3. Default threshold = 0.65 — never 0.5
4. Train on 400×400→256×256 — never 40×40 tiles
5. WeightedRandomSampler on train only
6. Differential LR: encoder 6e-5, decoder 6e-4
7. val_fp_rate_on_negatives logged every epoch
8. Stain normalization in prepare_data.py AND inference.py
9. set_seed(42) at top of train.py and evaluate.py
10. Mask values asserted as {0,1} float32 in dataset
11. CaVMamba encoder loaded with strict=False
12. Switch-UMamba always from scratch — no pretrained
13. Instance confidence = mean prob_map within instance pixels
14. Count = instances passing all filters
```
