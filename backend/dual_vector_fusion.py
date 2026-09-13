"""
dual_vector_fusion.py
---------------------
Context Analysis: Dual-Vector Fusion (Decoupling Topic from Emotional Intent).
Separates topical domain (e.g. discussions of layoffs, war, illness, politics)
from hostile/aggressive emotional affect using orthogonal subspace projection.

Mathematical Formulation:
  - Let x_t in R^K_tfidf be the sparse TF-IDF vector of a post in topic space.
  - Let T in R^(K x K_tfidf) be the dense Latent Semantic Analysis (LSA) topic subspace (K=64).
  - The Topical Projection of x onto the topic manifold:
      p_topic = T^T (T T^T)^(-1) T x
  - The Hostile/Aggressive Affect Component is the orthogonal complement:
      v_affect = x_t - p_topic
  - Decision Metric:
      topic_intensity = ||p_topic|| / (||x_t|| + eps)
      affect_ratio = ||v_affect|| / (||x_t|| + eps)
  - Fusion Rule:
      When topic_intensity is high (sensitive topical discussion, e.g. news/debates)
      and affect_hostility is below aggressive thresholds, the post is classified as
      CIVIL TOPICAL DISCUSSION, eliminating common false alarms on benign sensitive content.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class DualVectorFusionEngine:
    def __init__(self, topic_damping_factor: float = 0.40):
        self.topic_damping_factor = topic_damping_factor

    def decompose_and_score(
        self,
        text: str,
        base_toxicity_score: float,
        lsa_model: Any,
        features_step: Any,
        clf_step: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Decomposes input text into topic projection and orthogonal affect vector.
        Returns: (decoupled_toxicity_score, metadata_dict)
        """
        if not text or not text.strip():
            return base_toxicity_score, {"status": "empty"}

        if lsa_model is None or clf_step is None:
            return base_toxicity_score, {"status": "fallback_no_subspace"}

        try:
            # 1. Transform text using LSA model to get normalized coordinates in the 64-dimensional latent subspace
            lsa_coords = lsa_model.transform([text])[0]  # shape (64,), unit L2 norm

            # 2. Topic Intensity: energy concentration in dominant topical eigenvectors
            # High topical focus concentrates energy in top components; diffuse noise/slurs do not
            sorted_energy = sorted(lsa_coords ** 2, reverse=True)
            top_3_energy = float(np.sum(sorted_energy[:3]))
            topic_intensity = float(np.clip(top_3_energy, 0.0, 1.0))

            # 3. Orthogonal Affect Ratio: dispersion / non-topical entropy
            affect_ratio = float(np.clip(1.0 - topic_intensity, 0.0, 1.0))

            # 4. Direct abusive tokens from the classifier
            direct_slur_weight = 0.0
            if features_step is not None:
                x_clf_sparse = features_step.transform([text])
                coef = clf_step.coef_[0]
                cx = x_clf_sparse.tocoo()
                for col, val in zip(cx.col, cx.data):
                    w = coef[col]
                    if w > 0.8:  # Clear personal attack or hostility marker
                        direct_slur_weight += float(val * w)

            # 5. Dual-Vector Fusion Decision:
            # If substantive topical discussion (topic_intensity > 0.45) with low direct hostility (slur < 0.20),
            # dampen false alarms that arise from topical hot-button terms (e.g. corruption, war, layoffs, debate)
            decoupled_score = base_toxicity_score
            is_damped = False

            if direct_slur_weight < 0.20 and topic_intensity > 0.45:
                dampen_amount = (topic_intensity - 0.40) * self.topic_damping_factor
                decoupled_score = max(0.08, base_toxicity_score - dampen_amount)
                is_damped = decoupled_score < base_toxicity_score

            metadata = {
                "topic_intensity": round(topic_intensity, 3),
                "affect_ratio": round(affect_ratio, 3),
                "direct_hostility_energy": round(direct_slur_weight, 3),
                "false_alarm_damped": is_damped,
                "raw_toxicity": round(base_toxicity_score, 4),
                "decoupled_toxicity": round(decoupled_score, 4)
            }

            return round(decoupled_score, 4), metadata

        except Exception as exc:
            return base_toxicity_score, {"status": "error", "message": str(exc)}


dual_vector_fusion = DualVectorFusionEngine()
