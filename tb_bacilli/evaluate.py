"""
evaluate.py — Evaluation for TB Bacilli Segmentation.

Pixel metrics: Dice, IoU, Precision, Recall, F1
fp_rate_on_negatives: standalone metric

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
- Overlay: image + GT + predicted instances + confidence
- TP / FP / FN detection grids
- Negative image prediction grid
"""

import argparse
import json
import logging
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regionprops
from torch.cuda.amp import autocast

from augmentations import TTAWrapper, get_val_transforms
from config import TBConfig
from dataset import get_dataloaders
from instance_extraction import extract_instances_from_config
from models import get_model
from postprocess import create_detection_grid, create_overlay, format_results
from utils import (
    compute_fp_rate_on_negatives,
    compute_pixel_metrics,
    load_checkpoint,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def threshold_sweep(preds, targets, thresholds=None):
    """
    Sweep thresholds and compute metrics at each.

    Returns dict with arrays per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.96, 0.05)

    results = {
        "thresholds": thresholds.tolist(),
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "fp_rate_neg": [],
    }

    for thresh in thresholds:
        metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "f1": 0}
        for pred, target in zip(preds, targets):
            m = compute_pixel_metrics(pred, target, threshold=thresh)
            for k in metrics_sum:
                metrics_sum[k] += m[k]
        n = len(preds)
        for k in metrics_sum:
            results[k].append(metrics_sum[k] / n)

        fp_rate = compute_fp_rate_on_negatives(preds, targets, threshold=thresh)
        results["fp_rate_neg"].append(fp_rate)

    return results


def plot_threshold_curves(sweep_results, save_path):
    """Plot and save threshold sweep curves."""
    thresholds = sweep_results["thresholds"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Metrics vs threshold
    for key in ["dice", "iou", "precision", "recall", "f1"]:
        axes[0].plot(thresholds, sweep_results[key], label=key.upper(), marker="o", markersize=3)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Pixel Metrics vs Threshold")
    axes[0].legend()
    axes[0].grid(True)

    # FP rate on negatives
    axes[1].plot(
        thresholds, sweep_results["fp_rate_neg"], "r-o", markersize=3, label="FP Rate on Negatives"
    )
    axes[1].axhline(y=0.01, color="green", linestyle="--", label="1% Target")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("FP Rate")
    axes[1].set_title("FP Rate on Negative Images vs Threshold")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Threshold curves saved to {save_path}")


def compute_instance_metrics(pred_instances_list, gt_masks, iou_threshold=0.5):
    """
    Compute instance-level metrics.

    - Per-bacillus precision/recall (IoU>0.5 match)
    - Mean absolute count error
    - Count Pearson R correlation
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    pred_counts = []
    gt_counts = []

    for pred_instances, gt_mask in zip(pred_instances_list, gt_masks):
        # GT instances
        gt_labels = sk_label(gt_mask > 0)
        gt_regions = sk_regionprops(gt_labels)
        gt_count = len(gt_regions)
        pred_count = len(pred_instances)

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

        # Match
        matched_gt = set()
        tp = 0
        for inst in pred_instances:
            pred_mask = inst["mask"]
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_region in enumerate(gt_regions):
                if j in matched_gt:
                    continue
                gt_inst_mask = (gt_labels == gt_region.label).astype(np.uint8)
                intersection = np.sum(pred_mask * gt_inst_mask)
                union = np.sum(pred_mask) + np.sum(gt_inst_mask) - intersection
                iou = intersection / (union + 1e-7)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)

        fp = pred_count - tp
        fn = gt_count - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Instance precision/recall
    inst_precision = total_tp / (total_tp + total_fp + 1e-7)
    inst_recall = total_tp / (total_tp + total_fn + 1e-7)

    # Count metrics
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    mae = np.mean(np.abs(pred_counts - gt_counts))

    if len(pred_counts) > 1 and np.std(pred_counts) > 0 and np.std(gt_counts) > 0:
        count_r, _ = pearsonr(pred_counts, gt_counts)
    else:
        count_r = 0.0

    return {
        "instance_precision": float(inst_precision),
        "instance_recall": float(inst_recall),
        "mean_abs_count_error": float(mae),
        "count_pearson_r": float(count_r),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TB Bacilli Segmentation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument("--tta", action="store_true", help="Enable TTA")
    parser.add_argument("--save_vis", action="store_true", help="Save visualizations")
    args = parser.parse_args()

    config = TBConfig()

    # Set seed — NON-NEGOTIABLE
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(config)
    model = model.to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(config.checkpoint_dir, "best.pth")
    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model)
    else:
        logger.warning(f"No checkpoint found at {ckpt_path}")

    model.eval()

    # TTA
    tta_wrapper = None
    if args.tta or config.tta_enabled:
        tta_wrapper = TTAWrapper(model, device)

    # Data
    val_transforms = get_val_transforms(config.input_size)
    dataloaders = get_dataloaders(config, val_transforms=val_transforms)
    loader = dataloaders.get(args.split)
    if loader is None:
        logger.error(f"No data found for split '{args.split}'")
        return

    logger.info(f"Evaluating on {args.split} split...")

    # Collect predictions
    all_preds = []
    all_targets = []
    all_images = []
    all_instances = []

    with torch.no_grad():
        for images, masks, meta in loader:
            images_dev = images.to(device)

            if tta_wrapper:
                probs = tta_wrapper.predict(images_dev)
            else:
                if config.mixed_precision:
                    with autocast():
                        logits = model(images_dev)
                else:
                    logits = model(images_dev)
                probs = torch.sigmoid(logits)

            for i in range(images.size(0)):
                pred_np = probs[i, 0].cpu().numpy()
                target_np = masks[i, 0].numpy()
                img_np = images[i].permute(1, 2, 0).numpy()

                # Denormalize for visualization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = (img_np * std + mean) * 255
                img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

                all_preds.append(pred_np)
                all_targets.append(target_np)
                all_images.append(img_vis)

                # Instance extraction
                instances = extract_instances_from_config(pred_np, config)
                all_instances.append(instances)

    # ═══════════════════════════════════════════════════════
    # Pixel Metrics
    # ═══════════════════════════════════════════════════════

    # At default threshold
    metrics_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "f1": 0}
    for pred, target in zip(all_preds, all_targets):
        m = compute_pixel_metrics(pred, target, threshold=config.confidence_threshold)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
    n = len(all_preds)
    pixel_metrics = {k: v / n for k, v in metrics_sum.items()}

    fp_rate = compute_fp_rate_on_negatives(
        all_preds, all_targets, threshold=config.confidence_threshold
    )

    logger.info("=" * 60)
    logger.info(f"PIXEL METRICS (threshold={config.confidence_threshold})")
    for k, v in pixel_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"  fp_rate_on_negatives: {fp_rate:.4f}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Threshold Sweep
    # ═══════════════════════════════════════════════════════

    sweep = threshold_sweep(all_preds, all_targets)

    # Optimal threshold by max F1
    best_f1_idx = np.argmax(sweep["f1"])
    best_thresh = sweep["thresholds"][best_f1_idx]
    logger.info(f"Optimal threshold (max F1): {best_thresh:.2f} → F1={sweep['f1'][best_f1_idx]:.4f}")

    # Threshold where FP rate < 1%
    for i, (t, fp) in enumerate(zip(sweep["thresholds"], sweep["fp_rate_neg"])):
        if fp < 0.01:
            logger.info(f"FP rate < 1% at threshold: {t:.2f}")
            break

    plot_threshold_curves(
        sweep, os.path.join(config.output_dir, "threshold_curves.png")
    )

    # ═══════════════════════════════════════════════════════
    # Instance Metrics
    # ═══════════════════════════════════════════════════════

    inst_metrics = compute_instance_metrics(all_instances, all_targets)
    logger.info("=" * 60)
    logger.info("INSTANCE METRICS")
    for k, v in inst_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════

    results = {
        "pixel_metrics": pixel_metrics,
        "fp_rate_on_negatives": fp_rate,
        "optimal_threshold": best_thresh,
        "instance_metrics": inst_metrics,
        "threshold_sweep": sweep,
    }
    results_path = os.path.join(config.output_dir, f"eval_{args.split}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ═══════════════════════════════════════════════════════
    # Visualizations
    # ═══════════════════════════════════════════════════════

    if args.save_vis:
        vis_dir = os.path.join(config.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Sample overlays
        n_vis = min(20, len(all_images))
        for i in range(n_vis):
            overlay = create_overlay(all_images[i], all_targets[i], all_instances[i])
            cv2.imwrite(
                os.path.join(vis_dir, f"overlay_{i:04d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

            # Detection grids
            tp_grid, fp_grid, fn_grid = create_detection_grid(
                all_images[i], all_targets[i], all_instances[i]
            )
            cv2.imwrite(
                os.path.join(vis_dir, f"tp_{i:04d}.png"),
                cv2.cvtColor(tp_grid, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(vis_dir, f"fp_{i:04d}.png"),
                cv2.cvtColor(fp_grid, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(vis_dir, f"fn_{i:04d}.png"),
                cv2.cvtColor(fn_grid, cv2.COLOR_RGB2BGR),
            )

        # Negative image prediction grid
        neg_dir = os.path.join(vis_dir, "negatives")
        os.makedirs(neg_dir, exist_ok=True)
        neg_count = 0
        for i in range(len(all_targets)):
            if np.sum(all_targets[i]) == 0:  # negative image
                overlay = create_overlay(all_images[i], None, all_instances[i])
                cv2.imwrite(
                    os.path.join(neg_dir, f"neg_{neg_count:04d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                neg_count += 1
                if neg_count >= 20:
                    break

        logger.info(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
