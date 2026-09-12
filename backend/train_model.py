"""
train_model.py
--------------
Trains a lightweight, self-owned toxicity classifier (Word + Char TF-IDF + Logistic
Regression) on the expanded diverse dataset and saves it to `custom_toxic_model.joblib`.

Provides honest, rigorous before/after evaluation metrics (accuracy, F1, precision,
recall, confusion matrix), eliminating the artificial 100% synthetic overfitting.

Usage:
    python train_model.py
"""
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPANDED_DATA_PATH = os.path.join(DATA_DIR, "expanded_toxic_dataset.csv")
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "toxic_sample.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "custom_toxic_model.joblib"))
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.json")


def build_pipeline() -> Pipeline:
    """
    Constructs a dual-stream feature union (Word n-grams + Character n-grams)
    paired with balanced L2 Logistic Regression.
    - Word n-grams capture semantic phrases and syntax.
    - Character n-grams catch obfuscated leetspeak ('f*ck', 'k1ll', 'b!tch').
    """
    features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True
        )),
        ('char_tfidf', TfidfVectorizer(
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=6000,
            sublinear_tf=True
        ))
    ])

    return Pipeline([
        ('features', features),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.5,
            random_state=42
        ))
    ])


def train(data_path: str = None) -> dict:
    """Trains the classifier and writes model + metrics to disk."""
    if data_path is None:
        data_path = EXPANDED_DATA_PATH if os.path.exists(EXPANDED_DATA_PATH) else ORIGINAL_DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}.")

    start = time.time()

    df = pd.read_csv(data_path).dropna(subset=["text", "is_toxic"])
    df["is_toxic"] = df["is_toxic"].astype(int)

    X = df["text"].astype(str).values
    y = df["is_toxic"].values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist() # [[TN, FP], [FN, TP]]

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    elapsed = time.time() - start

    report_str = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print("=" * 65)
    print("Custom Toxicity Classifier — Training & Evaluation Complete")
    print("=" * 65)
    print(f"Dataset used: {os.path.basename(data_path)} ({len(df)} rows)")
    print(f"Train split: {len(X_train)} | Test split: {len(X_test)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f} (Binary) | {f1_macro:.4f} (Macro)")
    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(f"  True Negatives (Clean recognized):   {cm[0][0]}")
    print(f"  False Positives (Clean mistagged):   {cm[0][1]}")
    print(f"  False Negatives (Toxic missed):      {cm[1][0]}")
    print(f"  True Positives (Toxic caught):       {cm[1][1]}")
    print("\nDetailed Classification Report:\n" + report_str)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Training time: {elapsed:.2f}s")
    print("=" * 65)

    metrics = {
        "dataset": os.path.basename(data_path),
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm,
        "model_path": MODEL_PATH,
        "training_seconds": round(elapsed, 2),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    try:
        train()
        sys.exit(0)
    except Exception as exc:
        print(f"[train_model.py] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
