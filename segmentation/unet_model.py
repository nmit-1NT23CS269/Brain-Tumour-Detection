"""U-Net architecture, custom Dice+BCE loss, and metrics for binary brain tumour segmentation."""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np


def build_unet_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_filters: int = 32):
    """Build a 2D U-Net convolutional neural network for binary segmentation."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=input_shape, name="mri_input")

    # Encoder Block 1
    c1 = layers.Conv2D(num_filters, (3, 3), padding="same", name="enc_conv1_1")(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Conv2D(num_filters, (3, 3), padding="same", name="enc_conv1_2")(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Encoder Block 2
    c2 = layers.Conv2D(num_filters * 2, (3, 3), padding="same", name="enc_conv2_1")(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Conv2D(num_filters * 2, (3, 3), padding="same", name="enc_conv2_2")(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Encoder Block 3
    c3 = layers.Conv2D(num_filters * 4, (3, 3), padding="same", name="enc_conv3_1")(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Conv2D(num_filters * 4, (3, 3), padding="same", name="enc_conv3_2")(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = layers.Conv2D(num_filters * 8, (3, 3), padding="same", name="bottleneck_1")(p3)
    b = layers.BatchNormalization()(b)
    b = layers.Activation("relu")(b)
    b = layers.Conv2D(num_filters * 8, (3, 3), padding="same", name="bottleneck_2")(b)
    b = layers.BatchNormalization()(b)
    b = layers.Activation("relu")(b)

    # Decoder Block 3
    u3 = layers.Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(num_filters * 4, (3, 3), padding="same", name="dec_conv3_1")(u3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation("relu")(c4)
    c4 = layers.Conv2D(num_filters * 4, (3, 3), padding="same", name="dec_conv3_2")(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation("relu")(c4)

    # Decoder Block 2
    u2 = layers.Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(num_filters * 2, (3, 3), padding="same", name="dec_conv2_1")(u2)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation("relu")(c5)
    c5 = layers.Conv2D(num_filters * 2, (3, 3), padding="same", name="dec_conv2_2")(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation("relu")(c5)

    # Decoder Block 1
    u1 = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(num_filters, (3, 3), padding="same", name="dec_conv1_1")(u1)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation("relu")(c6)
    c6 = layers.Conv2D(num_filters, (3, 3), padding="same", name="dec_conv1_2")(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation("relu")(c6)

    # Output Binary Mask Probability
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", dtype="float32", name="mask_output")(c6)

    model = Model(inputs=inputs, outputs=outputs, name="NeuroScan_UNet")
    return model


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Compute Dice similarity coefficient metric."""
    import tensorflow as tf

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (total + smooth)


def iou_coef(y_true, y_pred, smooth: float = 1e-6):
    """Compute Intersection-over-Union (Jaccard Index) metric."""
    import tensorflow as tf

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


def dice_bce_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy and Dice loss for segmentation."""
    import tensorflow as tf

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss
