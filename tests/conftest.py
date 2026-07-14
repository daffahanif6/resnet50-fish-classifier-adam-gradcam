"""Shared pytest fixtures and configuration for the fish classifier tests.

Importing ``fish`` pulls in TensorFlow, so keep logging quiet and force a
non-interactive Matplotlib backend before any test module imports it.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

import fish


@pytest.fixture(scope="session")
def tiny_model():
    """A small ResNet-like functional model with a named final conv layer.

    It mirrors the interface the app relies on: ``model.inputs``,
    ``model.output``, ``model.get_layer(name)`` and a 9-class softmax head,
    while being tiny enough to train-free and fast for gradient tests.
    """
    inputs = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="conv5_block3_out")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(fish.CLASS_NAMES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def rgb_image():
    """A deterministic 32x48 RGB PIL image."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(48, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def small_img_array():
    """An (1, 8, 8, 3) float image batch matching ``tiny_model`` input."""
    rng = np.random.default_rng(1)
    return rng.random((1, 8, 8, 3)).astype("float32")
