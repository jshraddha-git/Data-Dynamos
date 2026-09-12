"""
train_model.py
----------------
Trains a lightweight, self-owned toxicity classifier on `data/toxic_sample.csv`
using a TF-IDF vectorizer + Logistic Regression pipeline, then persists the
fitted pipeline to `custom_toxic_model.joblib`.

This is the "primary" classifier referenced by main.py. It is intentionally
simple (TF-IDF + LogisticRegression) so it trains in seconds on a laptop CPU,
which matters for a live hackathon demo where judges may ask to see
`/train-custom-classifier` retrain the model on the spot.

Usage:
    python train_model.py
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Resolve paths relative to this file so the script works regardless of the
# working directory it's invoked from (important when main.py calls it via
# subprocess from the /train-custom-classifier endpoint).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "toxic_sample.csv")
MODEL_PATH = os.path.join(BASE_DIR, "custom_toxic_model.joblib")


def load_dataset(path: str) -> pd.DataFrame:
    """Load and lightly validate the training CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found at {path}. Expected columns: text, is_toxic"
        )
    df = pd.read_csv(path)
    if "text" not in df.columns or "is_toxic" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'is_toxic' columns.")
    df = df.dropna(subset=["text", "is_toxic"])
    df["text"] = df["text"].astype(str)
    df["is_toxic"] = df["is_toxic"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    """
    Build the TF-IDF + Logistic Regression pipeline.

    - TfidfVectorizer: unigrams + bigrams, English stopwords removed, capped
      vocabulary so the model stays small and fast to load.
    - LogisticRegression: 'liblinear' solver works well for small, sparse
      text-classification problems and trains near-instantly.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    max_features=5000,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_and_save(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    """Train the pipeline end-to-end and persist it. Returns eval metrics."""
    df = load_dataset(data_path)

    X = df["text"].tolist()
    y = df["is_toxic"].tolist()

    # Small holdout split purely for a sanity-check metric printed to stdout
    # and returned to the caller (used by the /train-custom-classifier
    # endpoint to show live accuracy during the demo).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)

    # Refit on the FULL dataset before saving so the deployed model benefits
    # from every labeled example, not just the training split.
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)

    print(f"[train_model] Trained on {len(X)} examples.")
    print(f"[train_model] Holdout accuracy: {acc:.4f}")
    print(report)
    print(f"[train_model] Model saved to: {model_path}")

    return {
        "n_samples": len(X),
        "holdout_accuracy": round(float(acc), 4),
        "model_path": model_path,
    }


if __name__ == "__main__":
    try:
        result = train_and_save()
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error surface
        print(f"[train_model] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
