"""
inference.py — Inference for TB Bacilli Segmentation.

MODE 1 — Small image (~400×400):
  Macenko normalize → resize 256 → model → TTA → instance extraction

MODE 2 — Large clinical image (>400×400):
  Macenko normalize → sliding window → overlap-averaged prob map
  → upsample → instance extraction

Batch inference: accept directory, auto-detect mode, output JSON + overlays.
"""

import argparse
import json
import logging
import os
import pickle
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast

from augmentations import TTAWrapper, get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from config import TBConfig
from instance_extraction import extract_instances_from_config
from models import get_model
from postprocess import create_overlay, format_results
from utils import MacenkoNormalizerNumpy, load_checkpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_stain_normalizer(path: str):
    """Load stain normalizer from disk."""
    if not os.path.isfile(path):
        logger.warning(f"Stain normalizer not found at {path}. Skipping normalization.")
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if data.get("type") == "staintools":
            return data["normalizer"]
        else:
            normalizer = MacenkoNormalizerNumpy()
            normalizer.stain_matrix_target = data["stain_matrix_target"]
            normalizer.maxC_target = data["maxC_target"]
            return normalizer
    return None


def preprocess_image(
    img: np.ndarray,
    normalizer=None,
    input_size: int = 256,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Preprocess single image for model input.

    Args:
        img: (H, W, 3) RGB uint8 image.
        normalizer: stain normalizer.
        input_size: resize target.

    Returns:
        (resized_img, input_tensor): processed image and model-ready tensor.
    """
    # Macenko normalize
    if normalizer is not None:
        try:
            img = normalizer.transform(img)
        except Exception as e:
            logger.warning(f"Stain normalization failed: {e}")

    # Resize
    resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Normalize (ImageNet)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).float().unsqueeze(0)

    return resized, tensor


def infer_small_image(
    img: np.ndarray,
    model: torch.nn.Module,
    config: TBConfig,
    normalizer=None,
    tta_wrapper: Optional[TTAWrapper] = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    MODE 1 — Small image (~400×400) inference.

    Returns:
        prob_map: (H, W) probability map at input_size.
        instances: list of instance dicts.
    """
    if device is None:
        device = next(model.parameters()).device

    _, input_tensor = preprocess_image(img, normalizer, config.input_size)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        if tta_wrapper:
            prob_map = tta_wrapper.predict(input_tensor)
        else:
            if config.mixed_precision:
                with autocast():
                    logits = model(input_tensor)
            else:
                logits = model(input_tensor)
            prob_map = torch.sigmoid(logits)

    prob_map = prob_map[0, 0].cpu().numpy()

    # Instance extraction
    instances = extract_instances_from_config(prob_map, config)

    return prob_map, instances


def infer_large_image(
    img: np.ndarray,
    model: torch.nn.Module,
    config: TBConfig,
    normalizer=None,
    tta_wrapper: Optional[TTAWrapper] = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    MODE 2 — Large clinical image (>400×400) inference.

    Sliding window with overlap averaging.

    Returns:
        prob_map: (H_orig, W_orig) probability map at original size.
        instances: list of instance dicts.
    """
    if device is None:
        device = next(model.parameters()).device

    H_orig, W_orig = img.shape[:2]

    # Macenko normalize
    if normalizer is not None:
        try:
            img = normalizer.transform(img)
        except Exception as e:
            logger.warning(f"Stain normalization failed: {e}")

    window_size = config.sliding_window_size
    stride = config.sliding_window_stride

    # Reflect padding
    pad_h = (window_size - H_orig % stride) % stride
    pad_w = (window_size - W_orig % stride) % stride
    img_padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    H_pad, W_pad = img_padded.shape[:2]

    # Accumulate probability map
    prob_accum = np.zeros((H_pad, W_pad), dtype=np.float64)
    count_accum = np.zeros((H_pad, W_pad), dtype=np.float64)

    model.eval()

    for y in range(0, H_pad - window_size + 1, stride):
        for x in range(0, W_pad - window_size + 1, stride):
            patch = img_padded[y : y + window_size, x : x + window_size]

            # Normalize
            tensor = patch.astype(np.float32) / 255.0
            tensor = (tensor - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            tensor = (
                torch.from_numpy(tensor.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                if tta_wrapper:
                    prob = tta_wrapper.predict(tensor)
                else:
                    if config.mixed_precision:
                        with autocast():
                            logits = model(tensor)
                    else:
                        logits = model(tensor)
                    prob = torch.sigmoid(logits)

            prob_np = prob[0, 0].cpu().numpy()
            prob_accum[y : y + window_size, x : x + window_size] += prob_np
            count_accum[y : y + window_size, x : x + window_size] += 1.0

    # Average
    count_accum = np.maximum(count_accum, 1.0)
    prob_map = prob_accum / count_accum

    # Crop to original size
    prob_map = prob_map[:H_orig, :W_orig]

    # Instance extraction on full-size map
    instances = extract_instances_from_config(prob_map, config)

    return prob_map, instances


def infer_single(
    img: np.ndarray,
    model: torch.nn.Module,
    config: TBConfig,
    normalizer=None,
    tta_wrapper: Optional[TTAWrapper] = None,
    device: torch.device = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Auto-detect mode and run inference.

    Small image (≤400×400): MODE 1
    Large image (>400×400): MODE 2
    """
    H, W = img.shape[:2]
    if max(H, W) > 400:
        logger.info(f"Large image ({H}×{W}) → sliding window mode")
        return infer_large_image(img, model, config, normalizer, tta_wrapper, device)
    else:
        logger.info(f"Small image ({H}×{W}) → direct mode")
        return infer_small_image(img, model, config, normalizer, tta_wrapper, device)


def batch_inference(
    image_dir: str,
    model: torch.nn.Module,
    config: TBConfig,
    output_dir: str,
    normalizer=None,
    tta_wrapper: Optional[TTAWrapper] = None,
    device: torch.device = None,
):
    """
    Batch inference on a directory of images.

    Outputs per image: mask PNG, overlay, JSON with instances and count.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    image_paths = sorted(
        glob(os.path.join(image_dir, "*"))
    )
    image_paths = [
        p for p in image_paths
        if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    logger.info(f"Batch inference on {len(image_paths)} images")

    all_results = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to read: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prob_map, instances = infer_single(
            img, model, config, normalizer, tta_wrapper, device
        )

        # Save mask
        mask_uint8 = (prob_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_dir, f"{basename}_mask.png"), mask_uint8)

        # Save overlay
        # Resize image to match prob_map if needed
        if img.shape[:2] != prob_map.shape:
            img_vis = cv2.resize(
                img, (prob_map.shape[1], prob_map.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            img_vis = img
        overlay = create_overlay(img_vis, None, instances)
        cv2.imwrite(
            os.path.join(overlay_dir, f"{basename}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        # Save JSON
        result = format_results(instances, img_path)
        with open(os.path.join(json_dir, f"{basename}.json"), "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)
        logger.info(f"  {basename}: {result['count']} instances detected")

    # Summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Batch inference complete. Summary: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TB Bacilli Inference")
    parser.add_argument("--input", type=str, required=True, help="Image path or directory")
    parser.add_argument("--output", type=str, default="./inference_output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--tta", action="store_true", help="Enable TTA")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    config = TBConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(config)
    model = model.to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(config.checkpoint_dir, "best.pth")
    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model)
    else:
        logger.warning(f"No checkpoint at {ckpt_path}. Using random weights.")

    model.eval()

    # Stain normalizer
    normalizer = load_stain_normalizer(config.stain_normalizer_path)

    # TTA
    tta_wrapper = None
    use_tta = config.tta_enabled if not args.no_tta else False
    if args.tta:
        use_tta = True
    if use_tta:
        tta_wrapper = TTAWrapper(model, device)

    if os.path.isdir(args.input):
        batch_inference(
            args.input, model, config, args.output, normalizer, tta_wrapper, device
        )
    else:
        img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Failed to read image: {args.input}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prob_map, instances = infer_single(
            img, model, config, normalizer, tta_wrapper, device
        )

        result = format_results(instances, args.input)
        logger.info(f"Detected {result['count']} bacilli instances")
        for inst in result["instances"]:
            logger.info(
                f"  Instance {inst['instance_id']}: "
                f"bbox={inst['bbox']}, conf={inst['confidence']:.4f}"
            )

        # Save
        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.input))[0]

        mask_uint8 = (prob_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f"{basename}_mask.png"), mask_uint8)

        with open(os.path.join(args.output, f"{basename}.json"), "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
