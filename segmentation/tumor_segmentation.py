"""Classical MRI tumour segmentation helpers."""

from __future__ import annotations

import cv2
import numpy as np


def segment_tumour_region(image_rgb: np.ndarray) -> dict:
    """Segment a probable tumour region using adaptive thresholding and contours."""
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
    }
