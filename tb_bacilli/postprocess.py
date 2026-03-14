"""
postprocess.py — Post-processing utilities for TB Bacilli Segmentation.

Includes visualization helpers, result formatting, and mask refinement.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Distinct colors for instance visualization
INSTANCE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
]


def create_overlay(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray],
    instances: List[Dict],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create visualization overlay.

    image + GT (green) + predicted instances (colored per instance)
    + confidence text on each bbox.

    Args:
        image: (H, W, 3) RGB image.
        gt_mask: (H, W) ground truth binary mask, or None.
        instances: list of instance dicts from extract_instances.
        alpha: overlay transparency.

    Returns:
        (H, W, 3) RGB overlay image.
    """
    overlay = image.copy()

    # Draw GT in green
    if gt_mask is not None:
        gt_overlay = np.zeros_like(overlay)
        gt_overlay[gt_mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 1.0, gt_overlay, alpha * 0.5, 0)

    # Draw instances
    for i, inst in enumerate(instances):
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        mask = inst["mask"]
        bbox = inst["bbox"]
        confidence = inst["confidence"]

        # Instance mask overlay
        inst_overlay = np.zeros_like(overlay)
        inst_overlay[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, inst_overlay, alpha, 0)

        # Bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        # Confidence text
        text = f"{confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay


def create_detection_grid(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_instances: List[Dict],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create TP / FP / FN detection grids.

    Args:
        image: (H, W, 3) RGB image.
        gt_mask: (H, W) ground truth mask.
        pred_instances: list of predicted instance dicts.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (tp_grid, fp_grid, fn_grid), each (H, W, 3) RGB.
    """
    H, W = gt_mask.shape[:2]

    # Get GT instances
    from skimage.measure import label as sk_label, regionprops as sk_regionprops

    gt_labels = sk_label(gt_mask > 0)
    gt_regions = sk_regionprops(gt_labels)

    # Match predictions to GT
    matched_gt = set()
    tp_grid = image.copy()
    fp_grid = image.copy()

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

        bbox = inst["bbox"]
        x1, y1, x2, y2 = bbox

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            cv2.rectangle(tp_grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(fp_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # FN: unmatched GT regions
    fn_grid = image.copy()
    for j, gt_region in enumerate(gt_regions):
        if j not in matched_gt:
            y1, x1, y2, x2 = gt_region.bbox
            cv2.rectangle(fn_grid, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return tp_grid, fp_grid, fn_grid


def format_results(instances: List[Dict], image_path: str = "") -> Dict:
    """
    Format instance results for JSON output.

    Removes non-serializable mask arrays, keeps bbox/confidence/area.
    """
    formatted_instances = []
    for inst in instances:
        formatted_instances.append(
            {
                "instance_id": inst["instance_id"],
                "bbox": inst["bbox"],
                "confidence": round(inst["confidence"], 4),
                "area": inst["area"],
                "eccentricity": round(inst["eccentricity"], 4),
            }
        )

    return {
        "image_path": image_path,
        "count": len(formatted_instances),
        "instances": formatted_instances,
    }


def refine_probability_map(
    prob_map: np.ndarray, threshold: float = 0.65
) -> np.ndarray:
    """
    Light refinement of probability map before instance extraction.

    Removes isolated small blobs below threshold.
    """
    binary = (prob_map >= threshold).astype(np.uint8)
    # Remove very small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    refined = prob_map.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 5:  # Very small noise
            refined[labels == i] = 0.0
    return refined
