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
