"""
instance_extraction.py — Instance extraction from probability maps.

Two-stage pipeline Stage 2:
Threshold → Morphological closing → Distance transform → Watershed →
Connected components → Per-instance metrics → Filtering → Sorting

No training involved — pure post-processing on model probability maps.
"""

import logging
from typing import Dict, List

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def extract_instances(
    prob_map: np.ndarray,
    confidence_threshold: float = 0.65,
    min_instance_area: int = 20,
    max_instance_area: int = 2000,
    eccentricity_threshold: float = 0.7,
    watershed_compactness: float = 0.001,
    watershed_min_distance: int = 5,
    morphological_kernel_size: int = 3,
) -> List[Dict]:
    """
    Extract bacillus instances from a probability map.

    Args:
        prob_map: (H, W) probability map from model, values in [0, 1].
        confidence_threshold: threshold for binarization (default 0.65).
        min_instance_area: minimum instance area in pixels (remove noise).
        max_instance_area: maximum instance area (remove non-bacilli).
        eccentricity_threshold: minimum eccentricity (bacilli are rods).
        watershed_compactness: watershed compactness parameter.
        watershed_min_distance: minimum distance between watershed seeds.
        morphological_kernel_size: kernel size for morphological closing.

    Returns:
        List of instance dicts, each with:
        {bbox, mask, confidence, area, eccentricity, instance_id}
        Sorted by confidence descending. count = len(list).
    """
    H, W = prob_map.shape

    # Step 1: Threshold → binary mask
    binary_mask = (prob_map >= confidence_threshold).astype(np.uint8)

    # Early return if no positive pixels
    if np.sum(binary_mask) == 0:
        return []

    # Step 2: Morphological closing (fill gaps along rod axis)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morphological_kernel_size, morphological_kernel_size)
    )
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: Distance transform
    dist_transform = ndimage.distance_transform_edt(binary_mask)

    # Step 4: Local maxima as watershed seeds
    if dist_transform.max() > 0:
        # Find local maxima
        local_max_coords = peak_local_max(
            dist_transform,
            min_distance=watershed_min_distance,
            labels=binary_mask,
        )
        # Create marker image
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        if len(local_max_coords) > 0:
            for i, (r, c) in enumerate(local_max_coords, start=1):
                markers[r, c] = i
        else:
            # Fallback: label connected components directly
            markers = label(binary_mask)
    else:
        markers = label(binary_mask)

    # Step 5: Watershed segmentation
    if markers.max() > 0 and np.any(markers):
        # Watershed needs -dist_transform as input (higher = lower elevation)
        labels_ws = watershed(
            -dist_transform,
            markers,
            mask=binary_mask,
            compactness=watershed_compactness,
        )
    else:
        labels_ws = label(binary_mask)

    # Step 6: Label connected components (clean up)
    if labels_ws.max() == 0:
        return []

    # Step 7: Per region compute properties
    regions = regionprops(labels_ws)
    instances = []

    for region in regions:
        area = region.area
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)

        # Convert to (x1, y1, x2, y2) format
        y1, x1, y2, x2 = bbox

        # Eccentricity (0 = circle, 1 = line)
        eccentricity = region.eccentricity

        # Instance mask
        instance_mask = (labels_ws == region.label).astype(np.uint8)

        # Confidence = mean probability within instance pixels
        instance_pixels = prob_map[instance_mask > 0]
        confidence = float(np.mean(instance_pixels)) if len(instance_pixels) > 0 else 0.0

        instances.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "mask": instance_mask,
                "confidence": confidence,
                "area": int(area),
                "eccentricity": float(eccentricity),
                "instance_id": int(region.label),
            }
        )

    # Step 8: Filter
    filtered = []
    for inst in instances:
        # Area filter
        if inst["area"] < min_instance_area:
            continue
        if inst["area"] > max_instance_area:
            continue
        # Eccentricity filter (bacilli are rods: high eccentricity)
        if inst["eccentricity"] < eccentricity_threshold:
            continue
        filtered.append(inst)

    # Step 9: Sort by confidence descending
    filtered.sort(key=lambda x: x["confidence"], reverse=True)

    return filtered


def extract_instances_from_config(prob_map: np.ndarray, config) -> List[Dict]:
    """
    Convenience wrapper that reads parameters from config.

    Args:
        prob_map: (H, W) probability map.
        config: TBConfig instance.

    Returns:
        List of instance dicts.
    """
    return extract_instances(
        prob_map,
        confidence_threshold=config.confidence_threshold,
        min_instance_area=config.min_instance_area,
        max_instance_area=config.max_instance_area,
        eccentricity_threshold=config.eccentricity_threshold,
        watershed_compactness=config.watershed_compactness,
        watershed_min_distance=config.watershed_min_distance,
        morphological_kernel_size=config.morphological_kernel_size,
    )
