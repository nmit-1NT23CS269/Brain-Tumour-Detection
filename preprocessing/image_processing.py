"""MRI preprocessing, validation, quality gating, and visualization pipeline."""

from __future__ import annotations

import io
from typing import Optional

import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from explainability.gradcam import generate_grad_cam
from utils import configure_logging

matplotlib.use("Agg")
LOGGER = configure_logging(__name__)
TARGET_SIZE = (224, 224)
MAX_FILE_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def validate_image(uploaded_file) -> tuple[bool, str]:
    """Validate file size and MIME type for uploads."""
    if uploaded_file is None:
        return False, "No file provided."
    if uploaded_file.type not in ALLOWED_TYPES:
        return False, f"Unsupported format '{uploaded_file.type}'. Please upload JPG or PNG."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_MB} MB."
    return True, "Valid"


def validate_mri_image_quality(img_array: np.ndarray) -> tuple[bool, str]:
    """Quality gate to reject non-brain MRI images (too small, blank, or saturated color photos)."""
    if img_array is None or img_array.size == 0:
        return False, "Empty or invalid image data."

    height, width = img_array.shape[:2]
    if height < 100 or width < 100:
        return False, f"Image dimensions ({width}x{height}) are too small for MRI diagnostic analysis (min 100x100)."

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if img_array.ndim == 3 else img_array
    var_intensity = float(np.var(gray))
    if var_intensity < 5.0:
        return False, "Image appears nearly blank or lacks sufficient contrast for MRI analysis."

    if img_array.ndim == 3 and img_array.shape[2] == 3:
        # MRI scans converted to RGB have near-identical R, G, B channels
        channel_diff = np.std(img_array, axis=2).mean()
        if channel_diff > 35.0:
            return False, "Uploaded image contains high color saturation. Please upload a standard grayscale Brain MRI scan."

    return True, "Valid MRI scan"


def reencode_uploaded_image(uploaded_file, output_format: str = "PNG") -> bytes:
    """Sanitize uploaded image file by stripping EXIF tags and re-drawing onto a clean canvas."""
    if hasattr(uploaded_file, "read"):
        content = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    else:
        content = uploaded_file

    with Image.open(io.BytesIO(content)) as img:
        img.verify()

    with Image.open(io.BytesIO(content)) as img:
        clean_img = Image.new("RGB", img.size)
        clean_img.paste(img.convert("RGB"))
        buffer = io.BytesIO()
        clean_img.save(buffer, format=output_format)
        return buffer.getvalue()


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Read uploaded file bytes into an RGB numpy array after EXIF stripping and re-encoding."""
    clean_bytes = reencode_uploaded_image(uploaded_file)
    return np.array(Image.open(io.BytesIO(clean_bytes)).convert("RGB"))


def resize_image(img: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize image to the standard model input size using bilinear interpolation."""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def apply_gaussian_blur(img: np.ndarray, ksize: tuple[int, int] = (3, 3)) -> np.ndarray:
    """Backward compatibility wrapper; returns original image without blurring mismatch."""
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Scale image into float32 [0, 1]."""
    return img.astype(np.float32) / 255.0


def to_model_input(img_normalized: np.ndarray) -> np.ndarray:
    """Add batch dimension for Keras models."""
    return np.expand_dims(img_normalized, axis=0)


def assess_mri_quality(img: np.ndarray) -> dict:
    """Estimate image quality from blur and contrast heuristics."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    if sharpness < 20 or contrast < 20:
        label = "Poor"
    elif sharpness < 60 or contrast < 35:
        label = "Fair"
    else:
        label = "Good"

    return {
        "label": label,
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
    }


def preprocess_array(img_array: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> dict:
    """Single source of truth preprocessing pipeline for in-memory RGB arrays."""
    rgb = img_array if img_array.ndim == 3 else np.stack([img_array] * 3, axis=-1)
    resized = resize_image(rgb, size=target_size)
    normalized = normalize_image(resized)
    quality_valid, quality_msg = validate_mri_image_quality(rgb)

    return {
        "original": rgb,
        "resized": resized,
        "blurred": resized,  # Kept for backward compatibility
        "normalized": normalized,
        "model_input": to_model_input(normalized),
        "quality": assess_mri_quality(resized),
        "is_valid_mri": quality_valid,
        "quality_message": quality_msg,
    }


def preprocess_pipeline(uploaded_file, target_size: tuple[int, int] = TARGET_SIZE) -> dict:
    """Process an uploaded file end to end for inference."""
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    result = preprocess_array(np.array(pil_img), target_size=target_size)
    result["file_bytes"] = raw_bytes
    return result


def preprocess_for_training(image_path: str) -> np.ndarray:
    """Load and preprocess an image from disk for training or offline analysis."""
    img = Image.open(image_path).convert("RGB")
    return preprocess_array(np.array(img))["normalized"]


def plot_preprocessing_steps(steps: dict) -> plt.Figure:
    """Render a 4-panel visualization of the preprocessing pipeline."""
    fig = plt.figure(figsize=(16, 4), facecolor="#0d1117")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    panels = [
        ("Original", steps["original"]),
        ("Resized (224x224)", steps["resized"]),
        ("Grayscale / Contrast", cv2.cvtColor(steps["resized"], cv2.COLOR_RGB2GRAY)),
        ("Normalized", (steps["normalized"] * 255).astype(np.uint8)),
    ]

    for idx, (title, img) in enumerate(panels):
        axis = fig.add_subplot(gs[idx])
        if img.ndim == 2:
            axis.imshow(img, cmap="gray")
        else:
            axis.imshow(img)
        axis.set_title(title, color="#58a6ff", fontsize=11, fontweight="bold", pad=8)
        axis.axis("off")
        if idx < len(panels) - 1:
            axis.annotate(
                "",
                xy=(1.12, 0.5),
                xycoords="axes fraction",
                xytext=(1.0, 0.5),
                arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=2),
            )

    plt.tight_layout()
    return fig


def plot_pixel_histogram(img_normalized: np.ndarray) -> plt.Figure:
    """Plot pixel intensity histograms for RGB channels."""
    fig, axis = plt.subplots(figsize=(6, 3), facecolor="#0d1117")
    for channel, color, label in zip(range(3), ["#ff6b6b", "#51cf66", "#339af0"], ["Red", "Green", "Blue"]):
        axis.hist(
            img_normalized[:, :, channel].ravel(),
            bins=64,
            color=color,
            alpha=0.6,
            label=label,
            density=True,
        )

    axis.set_facecolor("#161b22")
    axis.set_xlabel("Pixel Intensity", color="#8b949e")
    axis.set_ylabel("Density", color="#8b949e")
    axis.set_title("Pixel Intensity Distribution", color="#58a6ff", fontweight="bold")
    axis.tick_params(colors="#8b949e")
    axis.legend(facecolor="#21262d", labelcolor="#c9d1d9")
    for spine in axis.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout()
    return fig
