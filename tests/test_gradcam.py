"""Tests for the Grad-CAM heatmap generation and overlay helpers."""
import numpy as np
from PIL import Image

import fish


def test_make_gradcam_heatmap_shape_and_range(tiny_model, small_img_array):
    heatmap = fish.make_gradcam_heatmap(
        small_img_array, tiny_model, "conv5_block3_out"
    )
    # Matches the spatial dims of the named conv layer output.
    assert heatmap.shape == (8, 8)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_make_gradcam_heatmap_respects_pred_index(tiny_model, small_img_array):
    heatmap = fish.make_gradcam_heatmap(
        small_img_array, tiny_model, "conv5_block3_out", pred_index=2
    )
    assert heatmap.shape == (8, 8)
    assert np.isfinite(heatmap).all()


def test_display_gradcam_returns_image_matching_original_size(rgb_image):
    heatmap = np.linspace(0, 1, num=8 * 8, dtype="float32").reshape(8, 8)
    result = fish.display_gradcam(rgb_image, heatmap)
    assert isinstance(result, Image.Image)
    # Output must match the original image dimensions (width, height).
    assert result.size == rgb_image.size


def test_display_gradcam_alpha_changes_output(rgb_image):
    heatmap = np.ones((8, 8), dtype="float32")
    low = np.asarray(fish.display_gradcam(rgb_image, heatmap, alpha=0.1))
    high = np.asarray(fish.display_gradcam(rgb_image, heatmap, alpha=0.9))
    assert not np.array_equal(low, high)
