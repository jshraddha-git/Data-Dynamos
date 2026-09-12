"""
train_model.py
----------------
Trains a lightweight, self-owned toxicity classifier (TF-IDF + Logistic
Regression) on `data/toxic_sample.csv` and saves it to
`custom_toxic_model.joblib`.

This is the "custom_toxic_model" referenced by main.py as the PRIMARY
classifier. It exists so the team can demonstrate self-trained-model
capability live during judging (see the /train-custom-classifier endpoint
in main.py, which calls this script programmatically).

Usage:
    python train_model.py
"""
import os
import sys
import time

try:
    import joblib  # type: ignore[import-not-found]
except ImportError as exc:
    raise RuntimeError(
        "joblib is required to train and save the custom toxicity model. "
        "Install it with `pip install joblib`."
    ) from exc

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split # type: ignore[import-not-found]
from sklearn.pipeline import Pipeline # type: ignore[import-not-found]
from sklearn.metrics import classification_report, accuracy_score # type: ignore[import-not-found]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "toxic_sample.csv")
MODEL_PATH = os.path.join(BASE_DIR, "custom_toxic_model.joblib")


def train() -> dict:
    """Trains the classifier and writes it to disk. Returns a small metrics dict."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Make sure data/toxic_sample.csv exists."
        )

    start = time.time()

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "is_toxic"])
    df["is_toxic"] = df["is_toxic"].astype(int)

    X = df["text"].astype(str).values
    y = df["is_toxic"].values

    # Small dataset -> keep a modest test split so training set stays rich
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF vectorizer -> Logistic Regression classifier
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=2.0,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    joblib.dump(pipeline, MODEL_PATH)

    elapsed = time.time() - start

    print("=" * 60)
    print("Custom Toxicity Classifier — Training Complete")
    print("=" * 60)
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy on held-out test split: {accuracy:.3f}")
    print(report)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Training time: {elapsed:.2f}s")
    print("=" * 60)

    return {
        "accuracy": round(float(accuracy), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_path": MODEL_PATH,
        "training_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    try:
        metrics = train()
        sys.exit(0)
    except Exception as exc:  # pragma: no cover
        print(f"[train_model.py] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
