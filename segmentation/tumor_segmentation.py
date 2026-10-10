"""MRI tumour region highlight helpers for NeuroScan AI.

Integrates U-Net deep learning segmentation when available, falling back to adaptive
thresholding contour extraction ("Region highlight (heuristic)").

NON-CLINICAL DISCLAIMER:
------------------------
This module provides AI research visual overlays for educational and research visualization.
It is NOT intended for clinical diagnosis, lesion volumetric quantification, or surgical
boundary planning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import MODEL_EXPORT_DIR, configure_logging

LOGGER = configure_logging(__name__)
UNET_MODEL_PATH = MODEL_EXPORT_DIR / "unet_brain_mri.keras"
_CACHED_UNET = None
NON_CLINICAL_DISCLAIMER = (
    "This region highlight is an AI research visual overlay for educational visualization "
    "and is NOT intended for clinical diagnosis or surgical boundary planning."
)


def _load_unet_model():
    """Attempt loading exported U-Net model from disk."""
    global _CACHED_UNET
    if _CACHED_UNET is not None:
        return _CACHED_UNET

    if UNET_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            from segmentation.unet_model import dice_bce_loss, dice_coef, iou_coef

            LOGGER.info("Loading U-Net segmentation model from %s", UNET_MODEL_PATH)
            _CACHED_UNET = tf.keras.models.load_model(
                UNET_MODEL_PATH,
                custom_objects={
                    "dice_bce_loss": dice_bce_loss,
                    "dice_coef": dice_coef,
                    "iou_coef": iou_coef,
                },
            )
            return _CACHED_UNET
        except Exception as err:
            LOGGER.warning("Failed loading U-Net model (%s). Falling back to heuristic method.", err)

    return None


def _heuristic_region_highlight(image_rgb: np.ndarray) -> dict[str, Any]:
    """Fallback heuristic region highlight using adaptive thresholding and contours."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        2,
    )

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    overlay = image_rgb.copy()
    contour_count = 0
    largest_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        contour_count += 1
        largest_area = max(largest_area, float(area))
        cv2.drawContours(mask, [contour], -1, 255, -1)
        cv2.drawContours(overlay, [contour], -1, (255, 64, 64), 2)

    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    colored_mask[:, :, 1:] = 0
    blended = cv2.addWeighted(image_rgb, 0.78, colored_mask, 0.35, 0)
    blended = cv2.addWeighted(blended, 0.92, overlay, 0.08, 0)

    coverage = float(mask.sum() / 255.0 / mask.size)
    return {
        "mask": mask,
        "overlay": blended,
        "contour_overlay": overlay,
        "coverage_ratio": round(coverage, 4),
        "largest_area": round(largest_area, 2),
        "contour_count": contour_count,
        "method": "Region highlight (heuristic)",
        "disclaimer": NON_CLINICAL_DISCLAIMER,
    }


def segment_tumour_region(image_rgb: np.ndarray) -> dict[str, Any]:
    """Highlight probable tumour region using U-Net deep learning model or heuristic fallback."""
    unet_model = _load_unet_model()
    if unet_model is None:
        return _heuristic_region_highlight(image_rgb)

    try:
        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        input_tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        pred_prob = unet_model.predict(input_tensor, verbose=0)[0, :, :, 0]
        binary_mask_224 = (pred_prob >= 0.5).astype(np.uint8) * 255

        # Resize mask back to input dimensions
        mask = cv2.resize(binary_mask_224, (width, height), interpolation=cv2.INTER_NEAREST)

        # Create overlay
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        colored_mask[:, :, 1:] = 0  # Red highlight
        blended = cv2.addWeighted(image_rgb, 0.75, colored_mask, 0.35, 0)

        # Draw contour outlines
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = blended.copy()
        cv2.drawContours(overlay, contours, -1, (255, 64, 64), 2)

        coverage = float(mask.sum() / 255.0 / mask.size)
        largest_area = float(max([cv2.contourArea(c) for c in contours], default=0.0))

        return {
            "mask": mask,
            "overlay": overlay,
            "contour_overlay": overlay,
            "coverage_ratio": round(coverage, 4),
            "largest_area": round(largest_area, 2),
            "contour_count": len(contours),
            "method": "U-Net Deep Learning (Research Overlay)",
            "disclaimer": NON_CLINICAL_DISCLAIMER,
        }
    except Exception as err:
        LOGGER.warning("U-Net inference failed (%s). Falling back to heuristic region highlight.", err)
        return _heuristic_region_highlight(image_rgb)
