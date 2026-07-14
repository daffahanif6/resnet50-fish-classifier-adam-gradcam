"""Tests for the saliency-map fallback visualization helpers."""
import matplotlib.pyplot as plt
import numpy as np

import fish


def test_generate_saliency_map_shape_and_normalization(tiny_model, small_img_array):
    saliency = fish.generate_saliency_map(tiny_model, small_img_array)
    # One saliency value per input pixel location.
    assert saliency.shape == (8, 8)
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0 + 1e-6


def test_generate_saliency_map_is_finite(tiny_model, small_img_array):
    saliency = fish.generate_saliency_map(tiny_model, small_img_array)
    assert np.isfinite(saliency).all()


def test_display_saliency_map_returns_figure(rgb_image):
    saliency = np.random.default_rng(2).random((8, 8)).astype("float32")
    fig = fish.display_saliency_map(saliency, rgb_image)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)
