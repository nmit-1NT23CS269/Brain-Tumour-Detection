"""Out-of-distribution (OOD) guard for brain MRI images."""

from __future__ import annotations

import cv2
import numpy as np


def is_probable_brain_mri(
    img_array: np.ndarray,
    model_probs: np.ndarray | None = None,
    channel_diff_threshold: float = 30.0,
    min_dark_ratio: float = 0.12,
    max_dark_ratio: float = 0.96,
    max_entropy_threshold: float = 0.85,
    min_max_prob_threshold: float = 0.40,
) -> tuple[bool, str]:
    """Out-of-distribution (OOD) guard to detect non-brain MRI images.

    Checks image dimensions, aspect ratio, color channel saturation,
    dark background ratio, pixel histogram distribution/entropy, and model confidence/entropy.

    Args:
        img_array: RGB or grayscale image array of shape (H, W, C) or (H, W).
        model_probs: Optional model output probabilities array.
        channel_diff_threshold: Maximum allowable std dev across RGB channels.
        min_dark_ratio: Minimum fraction of dark background pixels required.
        max_dark_ratio: Maximum fraction of dark background pixels allowed.
        max_entropy_threshold: Maximum allowed normalized entropy for model outputs.
        min_max_prob_threshold: Minimum allowed maximum softmax probability.

    Returns:
        tuple[bool, str]: (is_valid, reason_message)
    """
    if img_array is None or img_array.size == 0:
        return False, "Empty or invalid image data."

    if img_array.ndim < 2:
        return False, "Invalid image tensor dimensions."

    height, width = img_array.shape[:2]
    if height < 100 or width < 100:
        return False, f"Image dimensions ({width}x{height}) are too small for brain MRI analysis (min 100x100)."

    # 1. Aspect Ratio Check (Brain MRIs are roughly square 0.5 <= aspect <= 2.0)
    aspect_ratio = height / float(width)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Brain MRI scans must be approximately square."

    # Convert to RGB and grayscale representations
    if img_array.ndim == 2:
        rgb = np.stack([img_array] * 3, axis=-1)
        gray = img_array
    else:
        rgb = img_array[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 2. Color Saturation / Channel Similarity Check
    # Grayscale MRIs converted to RGB have near-zero channel variance across R, G, B
    channel_std = float(np.std(rgb, axis=2).mean())
    if channel_std > channel_diff_threshold:
        return False, (
            f"Uploaded image contains high color saturation (channel std: {channel_std:.1f}). "
            "Please upload a standard grayscale Brain MRI scan, not a color photograph."
        )

    # 3. Contrast & Intensity Variance Check
    gray_var = float(np.var(gray))
    if gray_var < 8.0:
        return False, "Image appears nearly blank or lacks sufficient contrast for MRI analysis."

    # 4. Dark Background Ratio Check
    # Brain MRIs are surrounded by black/dark background (air)
    dark_pixel_ratio = float(np.mean(gray < 45))
    if dark_pixel_ratio < min_dark_ratio:
        return False, (
            f"Image lacks the characteristic dark background of a Brain MRI scan (dark ratio: {dark_pixel_ratio*100:.1f}%)."
        )
    if dark_pixel_ratio > max_dark_ratio:
        return False, "Image is predominantly black and does not contain visible intracranial structures."

    # 5. Spatial Texture / Noise Check (Random noise vs smooth anatomical scan)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist[hist > 0]
    hist_entropy = float(-np.sum(hist * np.log2(hist))) / np.log2(32)

    # Uniform random noise has histogram entropy near 1.0 and high Laplacian variance
    if hist_entropy > 0.94 and laplacian_var > 2500:
        return False, "Image histogram and texture match synthetic random noise rather than an MRI scan."

    # 6. Model Softmax Confidence & Entropy Check (if provided)
    if model_probs is not None:
        probs = np.asarray(model_probs, dtype=float).ravel()
        if probs.size > 0:
            probs = probs / np.sum(probs)
            max_p = float(np.max(probs))
            num_classes = len(probs)
            h = -np.sum(probs * np.log2(probs + 1e-12))
            norm_entropy = float(h / np.log2(num_classes)) if num_classes > 1 else 0.0

            if max_p < min_max_prob_threshold or norm_entropy > max_entropy_threshold:
                return False, (
                    f"Model classification uncertainty is too high (max probability {max_p*100:.1f}%, "
                    f"normalized entropy {norm_entropy:.2f}). Image is out-of-distribution."
                )

    return True, "Valid Brain MRI scan"
