"""
train_nsfw_model.py
--------------------
Trains a self-owned image NSFW classifier — LogisticRegression over the
hand-engineered features in image_features.py — with NO pretrained network
(no NudeNet, no torchvision weights, nothing downloaded).

Why synthetic training data: shipping or fetching a real nudity dataset
isn't appropriate for a public repo, and this project's hard constraint is
"models you trained yourself" — not "models trained on data we can't
legally/ethically distribute." So instead we procedurally generate labeled
images whose properties mirror the actual signal the feature extractor
looks for:
  - "explicit" class: large, smooth, skin-tone-dominant regions covering
    most of the frame (mimics close-up skin coverage).
  - "safe" class: varied multi-color/textured scenes, sometimes including
    a *small* skin-tone patch (mimics an ordinary photo with a person in it),
    so the model has to learn "skin coverage + smoothness", not just
    "any skin tone present" (which would make it useless).

This keeps the whole pipeline — data generation, feature extraction,
training, inference — self-authored and locally reproducible.

For a real deployment you'd swap `build_dataset()` for a loader over a
properly licensed, consent-based labeled dataset, but the model
architecture, training code, and serving code would not need to change.

Usage:
    python train_nsfw_model.py
"""

import os
import sys
import random

import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression  # type: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]
from sklearn.metrics import accuracy_score, classification_report  # type: ignore[reportMissingImports]
import joblib  # type: ignore[reportMissingImports]

from image_features import extract_image_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "nsfw_image_model.joblib")

random.seed(7)
np.random.seed(7)

SKIN_TONES = [
    (255, 224, 189),
    (241, 194, 125),
    (224, 172, 105),
    (198, 134, 66),
    (141, 85, 36),
    (255, 205, 148),
]
SCENE_COLORS = [
    (70, 130, 180),
    (34, 139, 34),
    (210, 180, 140),
    (105, 105, 105),
    (255, 99, 71),
    (30, 144, 255),
    (60, 179, 113),
    (255, 215, 0),
]


def _jitter(color, amount=10):
    return tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in color)


def make_synthetic_image(explicit: bool, size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), color=random.choice(SCENE_COLORS))
    draw = ImageDraw.Draw(img)

    if explicit:
        # Large, smooth, skin-tone-dominant region covering most of the frame.
        tone = random.choice(SKIN_TONES)
        coverage = random.uniform(0.78, 1.0)
        w, h = int(size * coverage), int(size * coverage)
        x0 = random.randint(0, max(0, size - w))
        y0 = random.randint(0, max(0, size - h))
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=tone)
        # A little smooth shading variation, but keep it low-texture.
        for _ in range(2):
            ex0, ey0 = random.randint(0, size), random.randint(0, size)
            ex1, ey1 = random.randint(0, size), random.randint(0, size)
            draw.ellipse(
                [min(ex0, ex1), min(ey0, ey1), max(ex0, ex1), max(ey0, ey1)],
                fill=_jitter(tone, amount=6),
            )
    else:
        # Varied, textured scene: multiple shapes/colors. Occasionally
        # include a small skin-tone patch, like a person in a normal photo.
        for _ in range(random.randint(5, 12)):
            palette = SCENE_COLORS + ([random.choice(SKIN_TONES)] if random.random() < 0.3 else [])
            color = random.choice(palette)
            x0, y0 = random.randint(0, size), random.randint(0, size)
            x1, y1 = random.randint(0, size), random.randint(0, size)
            box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            if random.random() < 0.5:
                draw.rectangle(box, fill=color)
            else:
                draw.ellipse(box, fill=color)

    return img


def build_dataset(n_per_class: int = 300):
    X, y = [], []
    for _ in range(n_per_class):
        X.append(extract_image_features(make_synthetic_image(explicit=True)))
        y.append(1)
    for _ in range(n_per_class):
        X.append(extract_image_features(make_synthetic_image(explicit=False)))
        y.append(0)
    return np.array(X), np.array(y)


def train_and_save(model_path: str = MODEL_PATH, n_per_class: int = 300) -> dict:
    X, y = build_dataset(n_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)

    # Refit on full dataset before saving.
    clf.fit(X, y)
    joblib.dump(clf, model_path)

    print(f"[train_nsfw_model] Trained on {len(X)} synthetic samples.")
    print(f"[train_nsfw_model] Holdout accuracy: {acc:.4f}")
    print(report)
    print(f"[train_nsfw_model] Model saved to: {model_path}")

    return {
        "n_samples": len(X),
        "holdout_accuracy": round(float(acc), 4),
        "model_path": model_path,
    }


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as exc:  # noqa: BLE001
        print(f"[train_nsfw_model] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
