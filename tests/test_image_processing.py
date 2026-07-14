"""Tests for image preprocessing and prediction helpers."""
import numpy as np
import pytest
from PIL import Image

import fish


def test_transform_image_returns_expected_shape(rgb_image):
    processed = fish.transform_image_for_prediction(rgb_image)
    assert processed.shape == (1, 224, 224, 3)


def test_transform_image_converts_non_rgb_modes():
    gray = Image.new("L", (10, 10), color=128)
    processed = fish.transform_image_for_prediction(gray)
    assert processed.shape == (1, 224, 224, 3)


def test_transform_image_does_not_mutate_original(rgb_image):
    original_size = rgb_image.size
    fish.transform_image_for_prediction(rgb_image)
    assert rgb_image.size == original_size


def test_transform_image_applies_resnet_preprocessing():
    # ResNet50 caffe-style preprocessing subtracts the ImageNet channel means
    # and reorders RGB -> BGR. A pure-white image therefore maps to fixed,
    # non-raw values rather than staying at 255.
    white = Image.new("RGB", (32, 32), color=(255, 255, 255))
    processed = fish.transform_image_for_prediction(white)
    expected_bgr = np.array([255 - 103.939, 255 - 116.779, 255 - 123.68])
    np.testing.assert_allclose(processed[0, 0, 0], expected_bgr, atol=1e-2)


class _FakeModel:
    """Minimal stand-in for a Keras model exposing ``predict``."""

    def __init__(self, prediction_row):
        self._row = np.asarray(prediction_row, dtype="float32")
        self.received = None

    def predict(self, x):
        self.received = x
        return np.array([self._row])


def test_predict_returns_top_3_sorted_descending(rgb_image):
    row = [0.01, 0.5, 0.02, 0.3, 0.03, 0.04, 0.05, 0.02, 0.03]
    model = _FakeModel(row)

    results = fish.predict(rgb_image, model)

    assert len(results) == 3
    labels = [label for label, _ in results]
    scores = [score for _, score in results]
    assert labels[0] == "Gilt-Head Bream"  # index 1 -> highest
    assert labels[1] == "Red Mullet"       # index 3 -> second
    assert scores == sorted(scores, reverse=True)


def test_predict_scores_are_python_floats(rgb_image):
    row = [0.0] * 9
    row[0] = 1.0
    results = fish.predict(rgb_image, _FakeModel(row))
    assert all(isinstance(score, float) for _, score in results)


def test_predict_feeds_preprocessed_batch_to_model(rgb_image):
    model = _FakeModel([0.1] * 9)
    fish.predict(rgb_image, model)
    assert model.received.shape == (1, 224, 224, 3)
