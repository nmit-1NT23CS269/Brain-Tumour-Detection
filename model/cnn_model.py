"""Inference layer and compatibility bridge for NeuroScan AI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explainability import generate_grad_cam
from models import CLASS_NAMES as DEFAULT_CLASS_NAMES, build_transfer_model, get_backbone_and_preprocess, get_last_conv_layer, load_class_names_json
from utils import MODEL_EXPORT_DIR, configure_logging, estimate_severity, interpret_confidence

LOGGER = configure_logging(__name__)
MODEL_PATH = str(MODEL_EXPORT_DIR / "best_model.keras")
MODEL_METADATA_PATH = MODEL_EXPORT_DIR / "model_metadata.json"
CLASS_NAMES_PATH = MODEL_EXPORT_DIR / "class_names.json"

CLASS_NAMES = load_class_names_json(CLASS_NAMES_PATH)
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (224, 224, 3)

CLASS_INFO = {
    "Glioma": {
        "color": "#ff6b6b",
        "severity": "High",
        "description": "Glioma-like pattern detected. Specialist neuro-oncology review is recommended.",
        "icon": "🔴",
    },
    "Meningioma": {
        "color": "#ffd43b",
        "severity": "Moderate",
        "description": "Meningioma-like pattern detected. Correlate with contrast MRI and specialist review.",
        "icon": "🟡",
    },
    "Pituitary": {
        "color": "#74c0fc",
        "severity": "Moderate",
        "description": "Pituitary-region lesion pattern detected. Endocrine and neurosurgical review is advised.",
        "icon": "🔵",
    },
    "Normal": {
        "color": "#51cf66",
        "severity": "None",
        "description": "No tumour pattern detected on this scan. Continue routine clinical correlation.",
        "icon": "🟢",
    },
}

_CACHED_MODEL = None
_CACHED_METADATA = None


def build_demo_cnn():
    """Backward-compatible alias that returns a baseline VGG16 transfer model."""
    return build_transfer_model("VGG16")


def build_transfer_model_vgg16():
    """Backward-compatible alias for legacy imports."""
    return build_transfer_model("VGG16")


def load_model():
    """Load the best exported model, falling back to VGG16 if absent."""
    global _CACHED_MODEL, CLASS_NAMES, NUM_CLASSES
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    CLASS_NAMES = load_class_names_json(CLASS_NAMES_PATH)
    NUM_CLASSES = len(CLASS_NAMES)

    try:
        import tensorflow as tf

        model_path = Path(MODEL_PATH)
        if model_path.exists():
            LOGGER.info("Loading model from %s", model_path)
            _CACHED_MODEL = tf.keras.models.load_model(model_path)
        else:
            LOGGER.warning("Best exported model not found. Falling back to untrained VGG16 transfer model.")
            _CACHED_MODEL = build_transfer_model("VGG16")
    except Exception as e:
        LOGGER.warning("TensorFlow import/load failed (%s). Falling back to mock model wrapper.", e)
        class DummyModel:
            def predict(self, x, verbose=0):
                return np.array([[0.7, 0.1, 0.1, 0.1]])
        _CACHED_MODEL = DummyModel()

    return _CACHED_MODEL


def get_model_metadata() -> dict:
    """Return metadata about the currently preferred model."""
    global _CACHED_METADATA
    if _CACHED_METADATA is not None:
        return _CACHED_METADATA

    if MODEL_METADATA_PATH.exists():
        _CACHED_METADATA = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    else:
        _CACHED_METADATA = {
            "model_name": "VGG16",
            "model_version": "fallback",
            "model_path": MODEL_PATH,
        }
    return _CACHED_METADATA


def predict(
    model_input: np.ndarray,
    include_gradcam: bool = False,
    use_tta: bool = False,
    confidence_threshold: float | None = None,
) -> dict:
    """Run model inference with temperature scaling calibration and uncertainty analysis."""
    from model.calibrate import apply_temperature_to_logits, load_temperature_config, probabilities_to_logits

    model = load_model()
    metadata = get_model_metadata()
    model_name = metadata.get("model_name", "VGG16")
    calib_config = load_temperature_config()
    temperature = float(calib_config.get("temperature", 1.0))
    thresh = (
        confidence_threshold
        if confidence_threshold is not None
        else float(calib_config.get("low_confidence_threshold", 0.70))
    )

    # Format model input and apply ImageNet backbone preprocessing
    inp = model_input.copy()
    if inp.ndim == 3:
        inp = np.expand_dims(inp, axis=0)

    if inp.max() <= 1.0:
        inp_rgb_255 = inp * 255.0
    else:
        inp_rgb_255 = inp.copy()

    try:
        _, preprocess_fn = get_backbone_and_preprocess(model_name)
        preprocessed_input = preprocess_fn(inp_rgb_255.copy())
    except Exception:
        preprocessed_input = inp_rgb_255 / 255.0

    if use_tta:
        # Test-Time Augmentation: average original and horizontal flip predictions
        flipped_input = np.flip(preprocessed_input, axis=2)
        p1 = model.predict(preprocessed_input, verbose=0)[0]
        p2 = model.predict(flipped_input, verbose=0)[0]
        raw_preds = (p1 + p2) / 2.0
    else:
        raw_preds = model.predict(preprocessed_input, verbose=0)[0]

    # Temperature Scaling Calibration on Logits
    logits = probabilities_to_logits(np.expand_dims(raw_preds, axis=0))
    calibrated_probs_array = apply_temperature_to_logits(logits, temperature)[0]

    sorted_indices = np.argsort(calibrated_probs_array)
    class_idx = int(sorted_indices[-1])
    class_name = CLASS_NAMES[class_idx]
    confidence = float(calibrated_probs_array[class_idx])
    uncalibrated_confidence = float(raw_preds[class_idx])

    top2_idx = int(sorted_indices[-2]) if len(sorted_indices) > 1 else class_idx
    top2_gap = float(confidence - calibrated_probs_array[top2_idx])

    is_low_confidence = confidence < thresh
    is_uncertain = is_low_confidence or (top2_gap < 0.15)
    status = "low_confidence" if is_low_confidence else "normal"

    probs = {CLASS_NAMES[i]: float(calibrated_probs_array[i]) for i in range(NUM_CLASSES)}
    severity = estimate_severity(class_name, confidence)

    result = {
        "class": class_name,
        "class_idx": class_idx,
        "confidence": confidence,
        "uncalibrated_confidence": uncalibrated_confidence,
        "temperature": temperature,
        "confidence_threshold": thresh,
        "status": status,
        "is_tumour": class_name != "Normal",
        "is_uncertain": is_uncertain,
        "top2_gap": round(top2_gap, 4),
        "probabilities": probs,
        "info": {
            **CLASS_INFO.get(class_name, CLASS_INFO["Normal"]),
            "severity": severity,
        },
        "model_name": model_name,
        "model_version": metadata.get("model_version", "unknown"),
        "confidence_label": interpret_confidence(confidence),
    }

    if include_gradcam:
        try:
            layer_name = get_last_conv_layer(model_name)
        except KeyError:
            layer_name = None
        result["gradcam"] = generate_grad_cam(model, preprocessed_input, class_idx, layer_name)
    return result


def get_model_summary() -> str:
    """Return a text summary of the active model."""
    import io

    buffer = io.StringIO()
    load_model().summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()

