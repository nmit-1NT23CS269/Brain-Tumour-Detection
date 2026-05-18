"""Multi-model registry for transfer-learning backbones."""

from __future__ import annotations

from dataclasses import dataclass


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


def build_transfer_model(model_name: str):
    """Build a transfer-learning classifier for one of the supported backbones."""
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    backbone_cls, _ = get_backbone_and_preprocess(model_name)
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    for layer in base.layers[:-20]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=base.input, outputs=outputs, name=f"NeuroScan_{model_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_last_conv_layer(model_name: str) -> str:
    """Return the preferred Grad-CAM layer for a model family."""
    return MODEL_SPECS[model_name].last_conv_hint
