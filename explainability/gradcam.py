"""Grad-CAM utilities and explainability metadata."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils import configure_logging


LOGGER = configure_logging(__name__)


def generate_grad_cam(
    model,
    img_array: np.ndarray,
    class_idx: int,
    last_conv_layer: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Generate a Grad-CAM heatmap blended with the model input."""
    try:
        import tensorflow as tf

        if last_conv_layer is None:
            conv_layers = [
                layer.name
                for layer in model.layers
                if "conv" in layer.name.lower() and hasattr(layer, "output")
            ]
            last_conv_layer = conv_layers[-1]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.squeeze(conv_outputs @ pooled_grads[..., tf.newaxis]).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8

        resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        heatmap_uint8 = np.uint8(255 * resized)
        heatmap_rgb = cv2.cvtColor(
            cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB,
        )
        source = np.uint8(img_array[0] * 255)
        return cv2.addWeighted(source, 0.6, heatmap_rgb, 0.4, 0)
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
