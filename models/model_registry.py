"""Multi-model registry for transfer-learning backbones."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES = ["Glioma", "Meningioma", "Normal", "Pituitary"]
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory_name: str
    preprocess_name: str
    last_conv_hint: str


MODEL_SPECS = {
    "VGG16": ModelSpec("VGG16", "VGG16", "vgg16", "block5_conv3"),
    "ResNet50": ModelSpec("ResNet50", "ResNet50", "resnet50", "conv5_block3_out"),
    "EfficientNetB0": ModelSpec("EfficientNetB0", "EfficientNetB0", "efficientnet", "top_conv"),
    "MobileNetV2": ModelSpec("MobileNetV2", "MobileNetV2", "mobilenet_v2", "Conv_1"),
}


def get_backbone_and_preprocess(model_name: str):
    """Resolve a Keras applications backbone class and preprocess function."""
    from tensorflow.keras.applications import (
        EfficientNetB0,
        MobileNetV2,
        ResNet50,
        VGG16,
        efficientnet,
        mobilenet_v2,
        resnet50,
        vgg16,
    )

    backbones = {
        "VGG16": (VGG16, vgg16.preprocess_input),
        "ResNet50": (ResNet50, resnet50.preprocess_input),
        "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input),
        "MobileNetV2": (MobileNetV2, mobilenet_v2.preprocess_input),
    }
    return backbones[model_name]


def save_class_names_json(export_dir: Path, class_names: list[str] = CLASS_NAMES) -> Path:
    """Save canonical class names and index mapping to JSON."""
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "class_names.json"
    data = {
        "class_names": class_names,
        "class_indices": {name: idx for idx, name in enumerate(class_names)},
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return json_path


def load_class_names_json(json_path: Path) -> list[str]:
    """Load canonical class names list from JSON."""
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data.get("class_names", CLASS_NAMES)
        except Exception:
            pass
    return CLASS_NAMES


def build_transfer_model(model_name: str, stage: str = "stage1", learning_rate: float = 1e-3):
    """Build a transfer-learning classifier for stage 1 (head training) or stage 2 (fine-tuning)."""
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    backbone_cls, _ = get_backbone_and_preprocess(model_name)
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    if stage == "stage1":
        base.trainable = False
    else:  # stage2 fine-tuning
        base.trainable = True
        # Keep BatchNorm layers frozen during fine-tuning to prevent statistic drift
        for layer in base.layers[:-35]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Dropout(0.40, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name=f"NeuroScan_{model_name}")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_stage2_model(model, learning_rate: float = 1e-4):
    """Unfreeze top layers of the stage 1 model for stage 2 fine-tuning while keeping BN frozen."""
    import tensorflow as tf
    from tensorflow.keras import layers

    # Make top layers trainable, keep earlier layers and all BN layers frozen
    for layer in model.layers[:-35]:
        layer.trainable = False
    for layer in model.layers[-35:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_last_conv_layer(model_name: str) -> str:
    """Return the preferred Grad-CAM layer for a model family."""
    return MODEL_SPECS.get(model_name, MODEL_SPECS["VGG16"]).last_conv_hint
