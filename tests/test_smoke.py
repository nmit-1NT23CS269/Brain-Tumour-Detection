"""Smoke test suite for NeuroScan AI pipeline, model inference, and authentication logic."""

from pathlib import Path
import numpy as np
from PIL import Image

from authentication.auth import hash_password, verify_password, validate_password
from model.cnn_model import CLASS_NAMES, predict
from preprocessing.image_processing import preprocess_array


def test_preprocessing_pipeline_on_sample_image():
    """Verify preprocessing pipeline runs on test_images/glioma.jpg without error."""
    image_path = Path(__file__).parent.parent / "test_images" / "glioma.jpg"
    assert image_path.exists(), f"Sample test image not found at {image_path}"

    pil_img = Image.open(image_path).convert("RGB")
    img_array = np.array(pil_img)

    processed = preprocess_array(img_array)
    assert "model_input" in processed
    assert processed["model_input"].shape == (1, 224, 224, 3)
    assert "quality" in processed
    assert processed["quality"]["label"] in {"Good", "Fair", "Poor"}


def test_predict_returns_valid_class_and_confidence():
    """Verify predict() returns one of the 4 valid class labels with confidence in [0, 1]."""
    image_path = Path(__file__).parent.parent / "test_images" / "glioma.jpg"
    pil_img = Image.open(image_path).convert("RGB")
    processed = preprocess_array(np.array(pil_img))

    result = predict(processed["model_input"])
    assert result["class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["probabilities"], dict)
    assert len(result["probabilities"]) == len(CLASS_NAMES)


def test_password_hashing_roundtrip():
    """Verify hash_password -> verify_password round trips correctly."""
    plain_password = "SecurePassword@123"
    hashed = hash_password(plain_password)

    assert hashed != plain_password
    assert verify_password(plain_password, hashed) is True
    assert verify_password("WrongPassword@123", hashed) is False


def test_validate_password_rules():
    """Verify validate_password accepts strong passwords and rejects weak ones."""
    # Strong password
    valid, msg = validate_password("StrongP@ssw0rd")
    assert valid is True
    assert msg == ""

    # Too short (< 8 chars)
    valid, msg = validate_password("P@ss1")
    assert valid is False

    # Missing uppercase
    valid, msg = validate_password("weakp@ssw0rd")
    assert valid is False

    # Missing digit
    valid, msg = validate_password("NoDigitsHere@")
    assert valid is False

    # Missing special character
    valid, msg = validate_password("NoSpecialChar123")
    assert valid is False
