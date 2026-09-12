"""
image_features.py
------------------
Hand-engineered, deterministic feature extraction for images — used by both
train_nsfw_model.py (to build the training set) and main.py (at inference
time), so training and serving stay consistent. This deliberately avoids any
pretrained neural network: it's plain pixel statistics computed with PIL/
NumPy, then classified by a self-trained scikit-learn model.

Features (8-dim vector):
  0. skin_ratio    - fraction of pixels in an HSV skin-tone band
  1-3. mean_r/g/b  - average channel intensities
  4-6. std_r/g/b   - channel intensity spread (flat color vs. varied scene)
  7. edge_density  - fraction of high-gradient pixels (texture/detail level)
"""

import numpy as np
from PIL import Image, ImageFilter

FEATURE_NAMES = [
    "skin_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "edge_density",
]

FEATURE_SIZE = (64, 64)
EDGE_THRESHOLD = 0.15


def extract_image_features(image: Image.Image) -> np.ndarray:
    """Compute the 8-dim feature vector for a PIL image."""
    image = image.convert("RGB").resize(FEATURE_SIZE)
    arr = np.asarray(image).astype(np.float32) / 255.0  # H x W x 3

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    std_r, std_g, std_b = float(r.std()), float(g.std()), float(b.std())

    # Simple HSV skin-tone band (a common lightweight heuristic — hue 0-50deg,
    # moderate saturation, decent brightness) used as a raw signal, NOT a
    # final verdict: the trained classifier learns how to weigh it alongside
    # the other features (e.g. skin tone + very low edge density + high
    # coverage reads very differently than skin tone in a normal photo).
    hsv = image.convert("HSV")
    hsv_arr = np.asarray(hsv).astype(np.float32)
    h = hsv_arr[:, :, 0] / 255.0 * 360.0
    s = hsv_arr[:, :, 1] / 255.0
    v = hsv_arr[:, :, 2] / 255.0
    skin_mask = (h >= 0) & (h <= 50) & (s >= 0.23) & (s <= 0.68) & (v >= 0.35)
    skin_ratio = float(skin_mask.mean())

    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges).astype(np.float32) / 255.0
    edge_density = float((edge_arr > EDGE_THRESHOLD).mean())

    return np.array(
        [skin_ratio, mean_r, mean_g, mean_b, std_r, std_g, std_b, edge_density],
        dtype=np.float32,
    )
