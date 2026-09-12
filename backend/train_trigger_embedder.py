"""
train_trigger_embedder.py
--------------------------
Trains a self-owned semantic embedding space for the trigger-topic filter,
replacing the pretrained sentence-transformers model entirely.

Approach: TF-IDF -> Truncated SVD (a.k.a. Latent Semantic Analysis). This is
a classic, fully-from-scratch way to learn a low-dimensional space where
topically-similar text ends up close together, fit on our own corpus with
no pretrained weights involved anywhere in the pipeline.

Training corpus = data/toxic_sample.csv (for general text variety) +
data/topic_corpus.csv (curated sentences spanning ~20 everyday topics —
politics, spoilers, crypto, sports, etc. — so common trigger words a user
might type have vocabulary coverage and land in a meaningful part of the
learned space).

At inference (see main.py: embed_texts / find_matching_trigger), both the
post text and each user trigger phrase are pushed through this same fitted
pipeline, then compared with cosine similarity.

Usage:
    python train_trigger_embedder.py
"""

import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOXIC_DATA_PATH = os.path.join(BASE_DIR, "data", "toxic_sample.csv")
TOPIC_CORPUS_PATH = os.path.join(BASE_DIR, "data", "topic_corpus.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trigger_embedder.joblib")


def load_corpus() -> list:
    texts = []
    if os.path.exists(TOXIC_DATA_PATH):
        df = pd.read_csv(TOXIC_DATA_PATH)
        texts.extend(df["text"].astype(str).tolist())
    if os.path.exists(TOPIC_CORPUS_PATH):
        df2 = pd.read_csv(TOPIC_CORPUS_PATH)
        texts.extend(df2["text"].astype(str).tolist())

    if len(texts) < 20:
        raise ValueError(
            "Not enough training text found. Ensure data/toxic_sample.csv "
            "and data/topic_corpus.csv exist."
        )
    return texts


def build_pipeline(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=8000,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "svd",
                TruncatedSVD(n_components=n_components, random_state=13),
            ),
        ]
    )


def train_and_save(model_path: str = MODEL_PATH) -> dict:
    texts = load_corpus()

    # Keep component count sane relative to corpus size to avoid a
    # degenerate/overfit SVD on very small corpora.
    n_components = max(2, min(120, len(texts) // 5))

    pipeline = build_pipeline(n_components=n_components)
    pipeline.fit(texts)

    joblib.dump(pipeline, model_path)

    explained = float(pipeline.named_steps["svd"].explained_variance_ratio_.sum())

    print(f"[train_trigger_embedder] Trained on {len(texts)} documents.")
    print(f"[train_trigger_embedder] Embedding dimensions: {n_components}")
    print(f"[train_trigger_embedder] Explained variance ratio (sum): {explained:.4f}")
    print(f"[train_trigger_embedder] Model saved to: {model_path}")

    return {
        "n_docs": len(texts),
        "n_components": n_components,
        "explained_variance": round(explained, 4),
        "model_path": model_path,
    }


if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as exc:  # noqa: BLE001
        print(f"[train_trigger_embedder] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)
