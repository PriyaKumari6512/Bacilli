"""
prepare_data.py — Data preparation for TB Bacilli Segmentation.

- Apply Macenko stain normalization to every image
- Fit normalizer on one reference training image
- Save fitted normalizer to ./artifacts/stain_normalizer.pkl
- If staintools unavailable → pure numpy Macenko fallback
- Resize images 400→256 INTER_LINEAR
- Resize masks 400→256 INTER_NEAREST (never smooth masks)
- Print class imbalance stats per split
- Save JSON manifest per split
- Idempotent — skip if outputs exist
"""

import argparse
import json
import logging
import os
import sys
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from config import TBConfig
from utils import MacenkoNormalizerNumpy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import staintools
try:
    import staintools

    HAS_STAINTOOLS = True
    logger.info("Using staintools for Macenko normalization.")
except ImportError:
    HAS_STAINTOOLS = False
    logger.info("staintools not available. Using pure numpy Macenko fallback.")


def get_image_mask_pairs(split_dir: str) -> List[Tuple[str, str]]:
    """
    Find image-mask pairs in a split directory.

    Expects structure:
    split_dir/
        images/  (or image/)
        masks/   (or mask/ or labels/ or label/)
    """
    # Try common directory names
    img_dirs = ["images", "image", "imgs", "img"]
    mask_dirs = ["masks", "mask", "labels", "label", "gt", "groundtruth"]

    img_dir = None
    mask_dir = None
    for d in img_dirs:
        candidate = os.path.join(split_dir, d)
        if os.path.isdir(candidate):
            img_dir = candidate
            break
    for d in mask_dirs:
        candidate = os.path.join(split_dir, d)
        if os.path.isdir(candidate):
            mask_dir = candidate
            break

    if img_dir is None or mask_dir is None:
        # Try flat structure
        logger.warning(
            f"Could not find image/mask subdirectories in {split_dir}. "
            f"Trying flat structure."
        )
        all_files = sorted(glob(os.path.join(split_dir, "*.*")))
        images = [f for f in all_files if "mask" not in f.lower() and "label" not in f.lower()]
        masks = [f for f in all_files if "mask" in f.lower() or "label" in f.lower()]
        return list(zip(images, masks))

    # Match by filename
    img_files = sorted(glob(os.path.join(img_dir, "*.*")))
    pairs = []
    for img_path in img_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # Try various mask extensions
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            mask_path = os.path.join(mask_dir, basename + ext)
            if os.path.isfile(mask_path):
                pairs.append((img_path, mask_path))
                break

    return pairs


def fit_stain_normalizer(reference_img: np.ndarray, config: TBConfig):
    """Fit stain normalizer and save to disk."""
    if HAS_STAINTOOLS:
        normalizer = staintools.StainNormalizer(method="macenko")
        # staintools expects uint8 RGB
        normalizer.fit(reference_img)
        # Wrap for saving
        import pickle

        os.makedirs(os.path.dirname(config.stain_normalizer_path), exist_ok=True)
        with open(config.stain_normalizer_path, "wb") as f:
            pickle.dump({"type": "staintools", "normalizer": normalizer}, f)
    else:
        normalizer = MacenkoNormalizerNumpy()
        normalizer.fit(reference_img)
        normalizer.save(config.stain_normalizer_path)

    logger.info(f"Stain normalizer fitted and saved to {config.stain_normalizer_path}")
    return normalizer


def normalize_image(img: np.ndarray, normalizer) -> np.ndarray:
    """Apply stain normalization to an image."""
    try:
        if HAS_STAINTOOLS and hasattr(normalizer, "transform"):
            if isinstance(normalizer, MacenkoNormalizerNumpy):
                return normalizer.transform(img)
            else:
                return normalizer.transform(img)
        elif isinstance(normalizer, MacenkoNormalizerNumpy):
            return normalizer.transform(img)
        elif isinstance(normalizer, dict) and normalizer.get("type") == "staintools":
            return normalizer["normalizer"].transform(img)
        else:
            return img
    except Exception as e:
        logger.warning(f"Stain normalization failed: {e}. Using original image.")
        return img


def process_split(
    split_name: str,
    split_dir: str,
    output_dir: str,
    manifest_dir: str,
    normalizer,
    input_size: int = 256,
):
    """Process a single data split."""
    out_img_dir = os.path.join(output_dir, split_name, "images")
    out_mask_dir = os.path.join(output_dir, split_name, "masks")
    manifest_path = os.path.join(manifest_dir, f"{split_name}.json")

    # Idempotent check
    if os.path.isfile(manifest_path):
        logger.info(f"Manifest exists for {split_name}: {manifest_path}. Skipping.")
        return

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    pairs = get_image_mask_pairs(split_dir)
    if not pairs:
        logger.warning(f"No image-mask pairs found in {split_dir}")
        return

    logger.info(f"Processing {split_name}: {len(pairs)} image-mask pairs")

    manifest = []
    total_pixels = 0
    total_bacilli_pixels = 0

    for img_path, mask_path in tqdm(pairs, desc=f"Processing {split_name}"):
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Read image and mask
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Failed to read mask: {mask_path}")
            continue

        # Stain normalization
        img = normalize_image(img, normalizer)

        # Resize: image with INTER_LINEAR, mask with INTER_NEAREST
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(
            mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST
        )

        # Binarize mask (ensure {0, 255} → {0, 1} after resize)
        mask = (mask > 127).astype(np.uint8) * 255

        # Save
        out_img_path = os.path.join(out_img_dir, f"{basename}.png")
        out_mask_path = os.path.join(out_mask_dir, f"{basename}.png")

        cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask_path, mask)

        # Stats
        bacilli_count = int(np.sum(mask > 0))
        pixel_count = mask.shape[0] * mask.shape[1]
        bacilli_ratio = bacilli_count / pixel_count if pixel_count > 0 else 0.0

        total_pixels += pixel_count
        total_bacilli_pixels += bacilli_count

        manifest.append(
            {
                "image_path": out_img_path,
                "mask_path": out_mask_path,
                "bacilli_pixel_count": bacilli_count,
                "bacilli_ratio": float(bacilli_ratio),
            }
        )

    # Save manifest
    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print stats
    overall_ratio = total_bacilli_pixels / total_pixels if total_pixels > 0 else 0.0
    pos_images = sum(1 for m in manifest if m["bacilli_ratio"] > 0)
    neg_images = len(manifest) - pos_images

    logger.info(f"{'='*50}")
    logger.info(f"Split: {split_name}")
    logger.info(f"  Total images: {len(manifest)}")
    logger.info(f"  Positive images (have bacilli): {pos_images}")
    logger.info(f"  Negative images (all background): {neg_images}")
    logger.info(f"  Overall bacilli pixel ratio: {overall_ratio:.4f} ({overall_ratio*100:.2f}%)")
    logger.info(f"  Manifest saved: {manifest_path}")
    logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DDS3 data")
    parser.add_argument("--config", type=str, default=None, help="Config overrides")
    args = parser.parse_args()

    config = TBConfig()
    logger.info("Starting data preparation...")

    # Find reference image for stain normalization
    train_pairs = get_image_mask_pairs(config.train_dir)
    if not train_pairs:
        logger.error(
            f"No training images found in {config.train_dir}. "
            f"Please check data directory structure."
        )
        sys.exit(1)

    # Use first training image as reference
    ref_img_path = config.stain_reference_path or train_pairs[0][0]
    logger.info(f"Using reference image for stain normalization: {ref_img_path}")
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    # Fit normalizer
    normalizer = fit_stain_normalizer(ref_img, config)

    # Process each split
    splits = {
        "train": config.train_dir,
        "val": config.val_dir,
        "test": config.test_dir,
    }

    for split_name, split_dir in splits.items():
        if os.path.isdir(split_dir):
            process_split(
                split_name,
                split_dir,
                config.processed_dir,
                config.manifest_dir,
                normalizer,
                config.input_size,
            )
        else:
            logger.warning(f"Split directory not found: {split_dir}")

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
