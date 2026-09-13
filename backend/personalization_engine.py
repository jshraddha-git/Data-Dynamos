"""
personalization_engine.py
-------------------------
Machine-Learned Adaptive Personal Sensitivity using Online SGD & Vector Shifts.
Learns from implicit/explicit user feedback (unhide / rehide actions) to customize
moderation sensitivity without leaking raw user history.

Mathematical Formulation:
  - Base Model: f_base(x) = sigma(x^T w_base)
  - Personal Vector Shift: w_user in R^D (initialized to 0)
  - Personalized Scoring: f_personal(x) = sigma(x^T w_base + x^T w_user)
  - Online Gradient Update on Feedback:
      unhide -> y = 0.0 (user tolerates this phrasing style)
      rehide -> y = 1.0 (user wants this content filtered)
      loss = BinaryCrossEntropy(f_personal(x), y)
      grad = (f_personal(x) - y) * x
      w_user <- (1 - lambda) * w_user - eta * grad
"""

import json
import os
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


import re

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "get", "got", "just", "like", "look", "make",
    "know", "think", "see", "come", "want", "give", "tell", "today", "yesterday",
    "tomorrow", "fellow", "check", "progress", "placed", "year", "years", "much",
    "many", "really", "will", "well", "also", "even", "back", "there", "good"
}


class OnlinePersonalizationEngine:
    def __init__(self, learning_rate: float = 0.04, l2_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self._user_cache: Dict[str, Dict[str, Any]] = {}

    def _get_user_file_path(self, client_id: str) -> str:
        safe_id = "".join(c for c in client_id if c.isalnum() or c in ("-", "_")).strip() or "default"
        return os.path.join(DATA_DIR, f"personal_model_{safe_id}.json")

    def _load_user_profile(self, client_id: str) -> Dict[str, Any]:
        if client_id in self._user_cache:
            return self._user_cache[client_id]

        file_path = self._get_user_file_path(client_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._user_cache[client_id] = data
                    return data
            except Exception:
                pass

        # New user profile
        new_profile = {
            "client_id": client_id,
            "weights": {},         # term -> weight shift (sparse vector)
            "unhide_count": 0,
            "rehide_count": 0,
            "total_feedback": 0,
            "learned_bias": 0.0,   # global personal bias shift
            "last_updated": time.time(),
        }
        self._user_cache[client_id] = new_profile
        return new_profile

    def _save_user_profile(self, client_id: str):
        profile = self._user_cache.get(client_id)
        if not profile:
            return
        file_path = self._get_user_file_path(client_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
        except Exception as exc:
            print(f"[personalization_engine] Error saving profile: {exc}")

    def score_post(self, text: str, base_score: float, client_id: str = "default") -> Tuple[float, float]:
        """
        Computes the personalized toxicity score by applying the user's learned vector shifts.
        Returns: (personalized_score, personal_shift_delta)
        """
        profile = self._load_user_profile(client_id)
        if profile.get("total_feedback", 0) == 0:
            return base_score, 0.0

        weights: Dict[str, float] = profile.get("weights", {})
        learned_bias: float = profile.get("learned_bias", 0.0)

        # Extract clean informative word tokens (length >= 4, non-stop-word)
        tokens = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))
        tokens = {t for t in tokens if t not in STOP_WORDS}
        shift = learned_bias

        for word, weight in weights.items():
            if word in tokens:
                shift += weight

        # Bound shift to prevent extreme runaway [-0.12, +0.12]
        bounded_shift = float(np.clip(shift, -0.12, 0.12))
        
        # Safety floor: if base_score is heavily toxic (>= 0.65), do NOT allow personal shift
        # to push it below 0.50 (safety guardrail against adversarial drift)
        if base_score >= 0.65 and (base_score + bounded_shift) < 0.50:
            bounded_shift = round(0.51 - base_score, 4)

        personalized_score = float(np.clip(base_score + bounded_shift, 0.0, 1.0))
        return round(personalized_score, 4), round(bounded_shift, 4)

    def record_feedback(
        self,
        text: str,
        action: str,  # "unhide" (label=0) or "rehide" (label=1)
        base_score: float,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Executes one step of Online SGD to update the user's vector weights.
        """
        profile = self._load_user_profile(client_id)
        target_y = 0.0 if action == "unhide" else 1.0

        current_score, current_shift = self.score_post(text, base_score, client_id)
        error = current_score - target_y  # gradient of binary cross-entropy: p - y

        # 1. Update global personal bias with tight bounds
        profile["learned_bias"] = float(np.clip(
            (1.0 - self.l2_reg) * profile.get("learned_bias", 0.0) - (self.learning_rate * 0.5) * error,
            -0.08, 0.08
        ))

        # 2. Extract active informative tokens (exclude stop words, min 4 chars)
        raw_tokens = re.findall(r"\b[a-z]{4,}\b", text.lower())
        informative_tokens = [w for w in raw_tokens if w not in STOP_WORDS][:15]
        weights = profile.setdefault("weights", {})

        for token in set(informative_tokens):
            w_curr = weights.get(token, 0.0)
            # SGD step with L2 decay
            w_next = (1.0 - self.l2_reg) * w_curr - self.learning_rate * error
            # Clamp per-term shift strictly to [-0.05, 0.05]
            weights[token] = round(float(np.clip(w_next, -0.05, 0.05)), 4)

        # Prune near-zero weights to keep model sparse & lightweight
        profile["weights"] = {k: v for k, v in weights.items() if abs(v) > 0.008}

        # Update stats
        if action == "unhide":
            profile["unhide_count"] = profile.get("unhide_count", 0) + 1
        else:
            profile["rehide_count"] = profile.get("rehide_count", 0) + 1
        profile["total_feedback"] = profile.get("total_feedback", 0) + 1
        profile["last_updated"] = time.time()

        self._save_user_profile(client_id)

        # Summarize top learned shifts
        sorted_weights = sorted(profile["weights"].items(), key=lambda x: x[1])
        top_tolerated = [k for k, v in sorted_weights[:3] if v < -0.01]
        top_strict = [k for k, v in sorted_weights[-3:] if v > 0.01]

        return {
            "status": "updated",
            "client_id": client_id,
            "action": action,
            "total_feedback": profile["total_feedback"],
            "learned_bias": round(profile["learned_bias"], 4),
            "active_feature_shifts": len(profile["weights"]),
            "top_tolerated_terms": top_tolerated,
            "top_filtered_terms": top_strict,
        }

    def get_user_stats(self, client_id: str = "default") -> Dict[str, Any]:
        profile = self._load_user_profile(client_id)
        weights = profile.get("weights", {})
        sorted_weights = sorted(weights.items(), key=lambda x: x[1])
        return {
            "total_feedback": profile.get("total_feedback", 0),
            "unhide_count": profile.get("unhide_count", 0),
            "rehide_count": profile.get("rehide_count", 0),
            "learned_bias": round(profile.get("learned_bias", 0.0), 4),
            "active_feature_shifts": len(weights),
            "top_tolerated": [{"term": k, "shift": v} for k, v in sorted_weights[:3] if v < 0],
            "top_filtered": [{"term": k, "shift": v} for k, v in sorted_weights[-3:] if v > 0],
        }

    def reset_user_profile(self, client_id: str = "default") -> Dict[str, Any]:
        """Resets the learned online weights and profile for the given client."""
        file_path = self._get_user_file_path(client_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        self._user_cache.pop(client_id, None)
        return self.get_user_stats(client_id)


personalization_engine = OnlinePersonalizationEngine()
