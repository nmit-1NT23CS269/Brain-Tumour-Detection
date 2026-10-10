"""Grad-CAM explainability generator and heatmap metadata helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from utils import configure_logging

LOGGER = configure_logging(__name__)
NON_CLINICAL_GRADCAM_DISCLAIMER = (
    "Grad-CAM heatmaps are AI research visual features for highlighting model attention regions "
    "and are NOT clinically validated diagnostic overlays."
)


def find_target_conv_layer(model: Any, layer_hint: Optional[str] = None) -> Any:
    """Recursively search for the target convolutional layer by name hint or layer type."""
    import tensorflow as tf

    if layer_hint:
        # 1. Search top-level model
        try:
            return model.get_layer(layer_hint)
        except (ValueError, KeyError):
            pass

        # 2. Search inside nested sub-models
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    return layer.get_layer(layer_hint)
                except (ValueError, KeyError):
                    pass

    # Fallback: Find the last 4D output Conv layer
    conv_candidates = []
    def _search_layers(layer_list):
        for layer in layer_list:
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                _search_layers(layer.layers)
            elif "conv" in layer.__class__.__name__.lower() or "conv" in layer.name.lower():
                try:
                    if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
                        conv_candidates.append(layer)
                    elif hasattr(layer, "output") and len(layer.output.shape) == 4:
                        conv_candidates.append(layer)
                except Exception:
                    pass

    _search_layers(model.layers)
    if conv_candidates:
        return conv_candidates[-1]

    raise ValueError("Could not locate a valid 4D convolutional layer for Grad-CAM.")


def generate_grad_cam(
    model: Any,
    img_array: np.ndarray,
    class_idx: int,
    last_conv_layer: Optional[str] = None,
    original_rgb: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Generate a Grad-CAM heatmap blended with the original scan image.

    Args:
        model: Keras classifier model.
        img_array: Preprocessed input tensor (batch shape 1xHxWxC or HxWxC).
        class_idx: Target class index to compute gradients for.
        last_conv_layer: Optional layer name hint.
        original_rgb: Optional clean un-preprocessed RGB image (HxWxC uint8).

    Returns:
        Optional[np.ndarray]: Blended RGB overlay uint8 array.
    """
    try:
        import tensorflow as tf

        # Ensure 4D batch shape (1, H, W, C)
        inp = img_array.copy()
        if inp.ndim == 3:
            inp = np.expand_dims(inp, axis=0)

        batch_size, height, width, channels = inp.shape

        target_layer = find_target_conv_layer(model, last_conv_layer)

        # Build sub-model extracting target conv output and final predictions
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(inp)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        # Weight conv outputs by pooled gradients
        heatmap = tf.squeeze(conv_outputs @ pooled_grads[..., tf.newaxis]).numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        # Resize heatmap to target spatial resolution (width, height)
        resized_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * resized_heatmap)
        heatmap_rgb = cv2.cvtColor(
            cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )

        # Prepare source RGB image for blending
        if original_rgb is not None:
            source = cv2.resize(original_rgb, (width, height))
            if source.ndim == 2:
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        else:
            # Reverse backbone scaling to reconstruct clean source RGB image
            raw_img = inp[0].copy()
            if raw_img.min() < 0:  # ImageNet mean-subtracted BGR (VGG16 / ResNet50)
                # BGR -> RGB mean restoration
                raw_img[:, :, 0] += 103.939
                raw_img[:, :, 1] += 116.779
                raw_img[:, :, 2] += 123.68
                raw_img = raw_img[:, :, ::-1]  # BGR to RGB
            elif raw_img.max() <= 1.0:
                raw_img = raw_img * 255.0
            source = np.clip(raw_img, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(source, 0.60, heatmap_rgb, 0.40, 0)
        return blended
    except Exception as exc:
        LOGGER.warning("Grad-CAM generation failed: %s", exc)
        return None


def summarize_heatmap_region(gradcam_img: Optional[np.ndarray]) -> dict[str, float] | None:
    """Estimate the center of activation from a Grad-CAM overlay."""
    if gradcam_img is None:
        return None

    red_channel = gradcam_img[:, :, 0].astype(np.float32)
    weights = red_channel / (red_channel.max() + 1e-8)
    total = float(weights.sum())
    if total <= 0:
        return None

    ys, xs = np.indices(weights.shape)
    center_y = float((ys * weights).sum() / total / weights.shape[0])
    center_x = float((xs * weights).sum() / total / weights.shape[1])
    return {"center_x": center_x, "center_y": center_y}
