"""Shared utilities for the fish classifier app.

Groups helpers that were previously duplicated across image preprocessing,
gradient-based visualization, and PNG serialization.
"""
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)


def image_to_array_batch(pil_img, size=IMG_SIZE):
    """Resizes a PIL image and returns it as a batched float array (1, H, W, 3)."""
    resized = pil_img.resize(size)
    img_array = tf.keras.utils.img_to_array(resized)
    return np.expand_dims(img_array, axis=0)


def squeeze_predictions(preds):
    """Returns the first element when a model yields a list of outputs."""
    if isinstance(preds, list):
        return preds[0]
    return preds


def normalize_01(tensor, relu=False):
    """Normalizes a tensor to the [0, 1] range with a numerical-stability epsilon.

    When ``relu`` is True the tensor is clamped at zero before scaling (Grad-CAM);
    otherwise it is min-shifted before scaling (saliency).
    """
    if relu:
        tensor = tf.maximum(tensor, 0)
    else:
        tensor = tensor - tf.reduce_min(tensor)
    return tensor / (tf.reduce_max(tensor) + 1e-9)


def figure_to_image(fig):
    """Renders a matplotlib figure to a PIL image via an in-memory PNG buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def pil_to_png_bytes(pil_img):
    """Serializes a PIL image to raw PNG bytes."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
