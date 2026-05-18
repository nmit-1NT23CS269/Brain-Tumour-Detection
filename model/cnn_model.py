"""Inference layer and compatibility bridge for NeuroScan AI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explainability import generate_grad_cam
from models import CLASS_NAMES, build_transfer_model, get_last_conv_layer
from utils import MODEL_EXPORT_DIR, configure_logging, estimate_severity, interpret_confidence


LOGGER = configure_logging(__name__)
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (224, 224, 3)
MODEL_PATH = str(MODEL_EXPORT_DIR / "best_model.keras")
MODEL_METADATA_PATH = MODEL_EXPORT_DIR / "model_metadata.json"

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
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    import tensorflow as tf

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        LOGGER.info("Loading model from %s", model_path)
        _CACHED_MODEL = tf.keras.models.load_model(model_path)
    else:
        LOGGER.warning("Best exported model not found. Falling back to untrained VGG16 transfer model.")
        _CACHED_MODEL = build_transfer_model("VGG16")
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


def predict(model_input: np.ndarray, include_gradcam: bool = False) -> dict:
    """Run model inference and optionally attach Grad-CAM output."""
    model = load_model()
    raw_preds = model.predict(model_input, verbose=0)[0]
    class_idx = int(np.argmax(raw_preds))
    class_name = CLASS_NAMES[class_idx]
    confidence = float(raw_preds[class_idx])
    probs = {CLASS_NAMES[i]: float(raw_preds[i]) for i in range(NUM_CLASSES)}
    severity = estimate_severity(class_name, confidence)
    metadata = get_model_metadata()

    result = {
        "class": class_name,
        "class_idx": class_idx,
        "confidence": confidence,
        "probabilities": probs,
        "is_tumour": class_name != "Normal",
        "info": {
            **CLASS_INFO[class_name],
            "severity": severity,
        },
        "model_name": metadata.get("model_name", "VGG16"),
        "model_version": metadata.get("model_version", "unknown"),
        "confidence_label": interpret_confidence(confidence),
    }

    if include_gradcam:
        try:
            layer_name = get_last_conv_layer(result["model_name"])
        except KeyError:
            layer_name = None
        result["gradcam"] = generate_grad_cam(model, model_input, class_idx, layer_name)
    return result


def get_model_summary() -> str:
    """Return a text summary of the active model."""
    import io

    buffer = io.StringIO()
    load_model().summary(print_fn=lambda line: buffer.write(line + "\n"))
    return buffer.getvalue()
