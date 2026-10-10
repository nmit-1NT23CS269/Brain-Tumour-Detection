"""Unit and accuracy sanity tests for NeuroScan AI model inference."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from model.cnn_model import CLASS_NAMES, predict, get_model_metadata, load_model
from preprocessing.image_processing import validate_mri_image_quality, preprocess_array


def test_model_metadata_valid():
    """Verify model metadata loads valid structure and model_name."""
    meta = get_model_metadata()
    assert isinstance(meta, dict)
    assert "model_name" in meta
    assert "model_path" in meta


def test_quality_gate_rejection():
    """Verify quality gate flags blank images and highly saturated color photos."""
    # Blank image (low variance)
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    valid_blank, msg_blank = validate_mri_image_quality(blank)
    assert not valid_blank
    assert "blank" in msg_blank.lower() or "contrast" in msg_blank.lower()

    # Too small image (< 100x100)
    small = np.ones((50, 50, 3), dtype=np.uint8) * 100
    valid_small, msg_small = validate_mri_image_quality(small)
    assert not valid_small
    assert "small" in msg_small.lower()

    # Saturated color image (non-MRI photo with contrast/texture)
    color_photo = np.random.randint(40, 220, size=(200, 200, 3), dtype=np.uint8)
    color_photo[:, :, 1] = 0
    color_photo[:, :, 2] = 0
    valid_color, msg_color = validate_mri_image_quality(color_photo)
    assert not valid_color
    assert "color saturation" in msg_color.lower() or "grayscale" in msg_color.lower()


def test_predict_schema_and_probability_sum():
    """Verify predict() returns complete result schema with normalized probabilities."""
    # Generate synthetic gray MRI-like scan with non-zero contrast
    synthetic_mri = np.random.randint(40, 200, size=(224, 224, 3), dtype=np.uint8)
    # Ensure R=G=B channels (grayscale)
    synthetic_mri[:, :, 1] = synthetic_mri[:, :, 0]
    synthetic_mri[:, :, 2] = synthetic_mri[:, :, 0]

    prep = preprocess_array(synthetic_mri)
    res = predict(prep["model_input"])

    assert "class" in res
    assert res["class"] in CLASS_NAMES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert isinstance(res["probabilities"], dict)
    assert len(res["probabilities"]) == len(CLASS_NAMES)
    assert "is_uncertain" in res
    assert "top2_gap" in res

    # Verify probability distribution sums to ~1.0
    prob_sum = sum(res["probabilities"].values())
    assert pytest.approx(prob_sum, abs=1e-3) == 1.0


def test_evaluation_metrics_computation():
    """Verify model.evaluate module metrics computation, ROC curves, and tumour recall highlights."""
    from model.evaluate import evaluate_metrics, compute_roc_data

    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    # Mock probabilities matrix
    y_probs = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.7, 0.1, 0.1],
        [0.05, 0.05, 0.85, 0.05],
        [0.1, 0.1, 0.1, 0.7],
        [0.75, 0.15, 0.05, 0.05],
        [0.2, 0.6, 0.1, 0.1],
        [0.05, 0.1, 0.8, 0.05],
        [0.05, 0.05, 0.1, 0.8],
    ])

    metrics = evaluate_metrics(y_true, y_probs, CLASS_NAMES)
    assert metrics["accuracy"] == 1.0
    assert "tumour_recalls" in metrics
    assert "Glioma" in metrics["tumour_recalls"]
    assert "Meningioma" in metrics["tumour_recalls"]
    assert "Pituitary" in metrics["tumour_recalls"]
    assert metrics["tumour_recalls"]["Glioma"] == 1.0

    roc = compute_roc_data(y_true, y_probs, CLASS_NAMES)
    assert roc["auc"]["macro"] == 1.0


def test_dedupe_check_logic():
    """Verify utils.dedupe_check module duplicate detection and leakage calculation."""
    from utils.dedupe_check import ImageRecord, find_duplicates, calculate_leakage_percentage
    import imagehash

    tr1 = ImageRecord(Path("train/g1.jpg"), "g1.jpg", "train", "Glioma", "md5_a", imagehash.hex_to_hash("0000000000000000"))
    tr2 = ImageRecord(Path("train/g2.jpg"), "g2.jpg", "train", "Glioma", "md5_b", imagehash.hex_to_hash("ffffffffffffffff"))

    te1 = ImageRecord(Path("test/g1_dup.jpg"), "g1_dup.jpg", "test", "Glioma", "md5_a", imagehash.hex_to_hash("0000000000000000"))
    te2 = ImageRecord(Path("test/g3_unique.jpg"), "g3_unique.jpg", "test", "Glioma", "md5_c", imagehash.hex_to_hash("1234567890abcdef"))

    dups = find_duplicates([tr1, tr2], [te1, te2], hamming_threshold=6)
    assert len(dups) == 1
    assert dups[0]["test_filename"] == "g1_dup.jpg"
    assert dups[0]["match_type"] == "exact_md5"

    leak_pct = calculate_leakage_percentage(dups, total_test_samples=2)
    assert pytest.approx(leak_pct) == 50.0


def test_temperature_scaling_calibration():
    """Verify temperature scaling, ECE computation, and low confidence status flagging."""
    from model.calibrate import compute_ece, fit_temperature, apply_temperature_to_logits, probabilities_to_logits

    y_true = np.array([0, 1, 0, 1, 0, 1])
    raw_probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.95, 0.05],
        [0.1, 0.9],
    ])

    ece, bin_stats = compute_ece(y_true, raw_probs, n_bins=5)
    assert 0.0 <= ece <= 1.0
    assert "bin_accuracies" in bin_stats

    logits = probabilities_to_logits(raw_probs)
    fitted_temp = fit_temperature(logits, y_true)
    assert 0.1 <= fitted_temp <= 10.0

    calibrated_probs = apply_temperature_to_logits(logits, fitted_temp)
    assert calibrated_probs.shape == raw_probs.shape
    assert pytest.approx(calibrated_probs.sum(axis=1), abs=1e-3) == np.ones(len(y_true))


def test_ood_guard_random_noise_color_photo_and_real_mri():
    """Verify is_probable_brain_mri rejects random noise & color photos while passing real MRI scans."""
    from preprocessing import is_probable_brain_mri
    from PIL import Image

    # 1. Random-noise image -> Should fail OOD check
    random_noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    is_valid_noise, msg_noise = is_probable_brain_mri(random_noise)
    assert is_valid_noise is False
    assert len(msg_noise) > 0

    # 2. Colorful photo (high R/G/B channel std) -> Should fail OOD check
    color_photo = np.zeros((224, 224, 3), dtype=np.uint8)
    color_photo[:, :, 0] = 230
    color_photo[:, :, 1] = 40
    color_photo[:, :, 2] = 80
    is_valid_photo, msg_photo = is_probable_brain_mri(color_photo)
    assert is_valid_photo is False
    assert "color saturation" in msg_photo.lower() or "grayscale" in msg_photo.lower()

    # 3. Real MRI sample from dataset or synthetic brain-like scan -> Should pass OOD check
    sample_mri_candidates = [
        Path(__file__).parent.parent / "data" / "splits" / "test" / "Glioma" / "Te-gl_1.jpg",
        Path(__file__).parent.parent / "data" / "raw" / "archive" / "Testing" / "glioma" / "Te-gl_1.jpg",
    ]
    real_mri_path = next((p for p in sample_mri_candidates if p.exists()), None)

    if real_mri_path:
        real_mri_img = np.array(Image.open(real_mri_path).convert("RGB"))
    else:
        # Create a synthetic brain MRI-like grayscale scan (black background with central brain tissue)
        real_mri_img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add central oval brain structure
        cv2.ellipse(real_mri_img, (112, 112), (70, 85), 0, 0, 360, (140, 140, 140), -1)
        cv2.circle(real_mri_img, (112, 112), 40, (180, 180, 180), -1)

    is_valid_mri, msg_mri = is_probable_brain_mri(real_mri_img)
    assert is_valid_mri is True
    assert "Valid" in msg_mri


def test_unet_segmentation_components_and_fallback():
    """Verify U-Net model building, custom loss/metrics, and region highlight fallback logic."""
    import cv2
    from segmentation import build_unet_model, dice_coef, iou_coef, dice_bce_loss, segment_tumour_region
    import tensorflow as tf

    # 1. Build U-Net model architecture
    model = build_unet_model(input_shape=(224, 224, 3), num_filters=16)
    assert model is not None
    assert model.output_shape == (None, 224, 224, 1)

    # 2. Test Dice and IoU metrics with synthetic tensors
    y_true = tf.constant([[[[1.0], [0.0]], [[1.0], [1.0]]]], dtype=tf.float32)
    y_pred = tf.constant([[[[1.0], [0.0]], [[1.0], [0.0]]]], dtype=tf.float32)

    dice_val = float(dice_coef(y_true, y_pred))
    iou_val = float(iou_coef(y_true, y_pred))
    loss_val = float(dice_bce_loss(y_true, y_pred))

    assert 0.0 <= dice_val <= 1.0
    assert 0.0 <= iou_val <= 1.0
    assert loss_val >= 0.0

    # 3. Test segment_tumour_region fallback output schema and non-clinical disclaimer
    synthetic_scan = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(synthetic_scan, (112, 112), 30, (200, 200, 200), -1)

    res = segment_tumour_region(synthetic_scan)
    assert "mask" in res
    assert "overlay" in res
    assert "method" in res
    assert "disclaimer" in res
    assert "Region highlight (heuristic)" in res["method"] or "U-Net" in res["method"]
    assert "NOT intended for clinical diagnosis" in res["disclaimer"]






