"""
train_lsa_semantic.py
---------------------
Trains a 100% self-owned Latent Semantic Analysis (LSA) model using
TF-IDF + TruncatedSVD (Singular Value Decomposition) to provide fast,
local semantic embeddings for trigger-topic filtering without relying on
pretrained SentenceTransformers or external deep learning weights.

Pipeline:
  Text -> TfidfVectorizer -> TruncatedSVD (128 dims) -> Normalizer -> Dense Semantic Vector

Usage:
  python train_lsa_semantic.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "custom_semantic_lsa.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "expanded_toxic_dataset.csv")

def get_training_corpus():
    """Compiles a diverse vocabulary corpus across common social topics and domains."""
    texts = []
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        texts.extend(df["text"].dropna().astype(str).tolist())
    
    # Domain-specific trigger vocabulary expansion
    semantic_anchors = [
        # Spoilers / Entertainment
        "major movie spoiler character death ending revealed plot twist season finale leak screenplay",
        "who died in the season finale climax unexpected betrayal revealed secret identity post-credits scene",
        
        # Layoffs / Economy / Career
        "massive tech layoffs severance package workforce reduction job cuts corporate downsizing unemployment hiring freeze",
        "company firing thousands of engineers restructuring division laid off recession economic downturn",
        
        # Crypto / Financial scams
        "cryptocurrency bitcoin altcoin pump and dump rug pull guaranteed returns get rich quick forextrading web3 nft mint",
        "double your money in crypto send eth to address binary options investment scheme telegram signal group",
        
        # Politics / Elections
        "presidential election ballot voting results corrupt politicians senate filibuster partisan campaign rally poll",
        "political scandal partisan propaganda government coverup conspiracy legislation congress impeachment",
        
        # Self-harm / Mental distress
        "depression suicidal thoughts self harm despair hopelessness cannot go on ending my life loneliness sorrow",
        "overwhelmed by anxiety panic attack mental health breakdown crying every night clinical depression",
        
        # Fitness / Diet culture
        "extreme calorie deficit rapid weight loss fasting starvation body dysmorphia eating disorder purging skinny",
        "body shaming unrealistic fitness standards workout exhaustion crash diet body transformation",
    ]
    texts.extend(semantic_anchors)
    return texts

def train_lsa_model():
    start = time.time()
    corpus = get_training_corpus()
    print(f"[train_lsa_semantic.py] Fitting LSA Semantic Model on {len(corpus)} documents...")

    # Build Pipeline: TF-IDF -> TruncatedSVD -> L2 Normalizer
    lsa_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_features=12000,
            sublinear_tf=True
        )),
        ('svd', TruncatedSVD(
            n_components=64, # 64 latent semantic dimensions
            algorithm='randomized',
            n_iter=10,
            random_state=42
        )),
        ('norm', Normalizer(norm='l2', copy=False))
    ])

    lsa_pipeline.fit(corpus)
    joblib.dump(lsa_pipeline, MODEL_PATH)
    elapsed = time.time() - start

    # Quick semantic test
    test_queries = ["layoffs", "spoilers", "crypto"]
    test_docs = [
        "Google announced another round of severance packages and job cuts today.",
        "The post-credits scene completely ruined the ending of the film for me.",
        "Guaranteed 500% profit when you buy this new memecoin token on Uniswap!",
        "Beautiful sunrise walk through the park with my golden retriever."
    ]

    q_vecs = lsa_pipeline.transform(test_queries)
    d_vecs = lsa_pipeline.transform(test_docs)
    sims = cosine_similarity(q_vecs, d_vecs)

    print("=" * 60)
    print("Self-Trained LSA Semantic Model — Training Complete")
    print("=" * 60)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Training time: {elapsed:.2f}s")
    print("Validation Similarity Matrix:")
    for i, q in enumerate(test_queries):
        top_doc_idx = int(np.argmax(sims[i]))
        print(f"  Trigger: '{q}' -> Best match: '{test_docs[top_doc_idx][:40]}...' (Score: {sims[i][top_doc_idx]:.3f})")
    print("=" * 60)

    return {
        "status": "success",
        "model_path": MODEL_PATH,
        "elapsed_seconds": round(elapsed, 2)
    }

if __name__ == "__main__":
    train_lsa_model()
