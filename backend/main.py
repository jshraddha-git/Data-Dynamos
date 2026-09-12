"""
main.py
--------
FastAPI backend for the AI-Powered Wellbeing Buffer.

Endpoints:
    POST /analyze-post              -> toxicity + NSFW + semantic trigger check
    POST /check-draft                -> pre-post toxicity check + rephrase suggestion
    POST /train-custom-classifier    -> live retrain of the self-owned toxicity model
    POST /calculate-mood-impact      -> session mood/wellbeing scoring
    GET  /health                     -> quick status check for the extension popup

Model precedence (toxicity):
    1. PRIMARY   -> custom_toxic_model.joblib (TF-IDF + LogisticRegression,
                    trained by this team via train_model.py) — proves
                    self-trained-model capability.
    2. FALLBACK  -> Detoxify('original-small'), used only if the custom
                    model file is missing or fails to load.

No commercial LLM APIs (OpenAI/Anthropic/Gemini) are used anywhere in this
file. All inference is local/self-hosted.
"""

import io
import os
import re
import subprocess
import sys
from typing import List, Optional

import joblib
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "custom_toxic_model.joblib")
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "train_model.py")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Wellbeing Buffer — Backend",
    description="Local ML backend for pre-render content filtering.",
    version="1.0.0",
)

# The Chrome extension calls this API from content-script/popup contexts,
# which run against arbitrary page origins -> allow all origins for the
# hackathon prototype. Lock this down to your extension's origin in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_custom_toxic_model = None            # sklearn Pipeline (primary)
_detoxify_model = None                # Detoxify instance (fallback)
_sentence_model = None                # SentenceTransformer for semantic triggers
_nsfw_image_model = None              # NudeNet detector

_using_fallback_toxicity = False

# Simple explicit-language wordlist for text-based NSFW pattern matching.
# Intentionally coarse (word-boundary regex) — this supplements the model
# score, it isn't the sole signal.
EXPLICIT_TEXT_PATTERNS = [
    r"\bnsfw\b",
    r"\bxxx\b",
    r"\bexplicit content\b",
    r"\bnude[sz]?\b",
    r"\bporn\w*\b",
]
_EXPLICIT_TEXT_REGEX = re.compile("|".join(EXPLICIT_TEXT_PATTERNS), re.IGNORECASE)


def _load_custom_toxic_model():
    """Attempts to load the team's self-trained joblib model."""
    global _custom_toxic_model
    if _custom_toxic_model is not None:
        return _custom_toxic_model
    if os.path.exists(MODEL_PATH):
        try:
            _custom_toxic_model = joblib.load(MODEL_PATH)
            print(f"[main.py] Loaded custom toxicity model from {MODEL_PATH}")
        except Exception as exc:
            print(f"[main.py] Failed to load custom model ({exc}); will use fallback.")
            _custom_toxic_model = None
    return _custom_toxic_model


def _load_detoxify_fallback():
    """Lazily loads Detoxify('original-small') as the fallback toxicity model."""
    global _detoxify_model
    if _detoxify_model is not None:
        return _detoxify_model
    try:
        from detoxify import Detoxify

        _detoxify_model = Detoxify("original-small")
        print("[main.py] Loaded Detoxify('original-small') fallback model.")
    except Exception as exc:
        print(f"[main.py] Could not load Detoxify fallback ({exc}).")
        _detoxify_model = None
    return _detoxify_model


def _load_sentence_model():
    """Lazily loads the sentence-transformer used for semantic trigger matching."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[main.py] Loaded SentenceTransformer('all-MiniLM-L6-v2').")
    except Exception as exc:
        print(f"[main.py] Could not load sentence-transformer ({exc}).")
        _sentence_model = None
    return _sentence_model


def _load_nsfw_image_model():
    """Lazily loads the NudeNet image classifier for NSFW image tagging."""
    global _nsfw_image_model
    if _nsfw_image_model is not None:
        return _nsfw_image_model
    try:
        from nudenet import NudeClassifier

        _nsfw_image_model = NudeClassifier()
        print("[main.py] Loaded NudeNet NudeClassifier.")
    except Exception as exc:
        print(f"[main.py] Could not load NudeNet ({exc}).")
        _nsfw_image_model = None
    return _nsfw_image_model


@app.on_event("startup")
def on_startup():
    """Warm-load the primary model at startup; fallback is loaded lazily on first use."""
    global _using_fallback_toxicity
    model = _load_custom_toxic_model()
    _using_fallback_toxicity = model is None
    if _using_fallback_toxicity:
        print("[main.py] custom_toxic_model.joblib not found -> "
              "will use Detoxify fallback on first request. "
              "Run `python train_model.py` (or call /train-custom-classifier) "
              "to enable the primary self-trained model.")


# ---------------------------------------------------------------------------
# Core toxicity scoring (shared by /analyze-post and /check-draft)
# ---------------------------------------------------------------------------
def score_toxicity(text: str) -> float:
    """
    Returns a toxicity probability in [0, 1].
    Tries the primary custom model first; falls back to Detoxify.
    """
    global _using_fallback_toxicity

    text = (text or "").strip()
    if not text:
        return 0.0

    model = _load_custom_toxic_model()
    if model is not None:
        try:
            proba = model.predict_proba([text])[0]
            # class order follows the fitted labels_; find index of class "1"
            classes = list(model.classes_)
            idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
            _using_fallback_toxicity = False
            return float(proba[idx])
        except Exception as exc:
            print(f"[main.py] Primary model inference failed ({exc}); falling back.")

    # Fallback path
    detox = _load_detoxify_fallback()
    if detox is not None:
        try:
            results = detox.predict(text)
            _using_fallback_toxicity = True
            return float(results.get("toxicity", 0.0))
        except Exception as exc:
            print(f"[main.py] Detoxify inference failed ({exc}).")

    # Last-resort heuristic if neither model is available (keeps API usable
    # in a bare-bones dev environment with no model files/deps yet).
    crude_flags = ["idiot", "stupid", "hate you", "kill yourself", "pathetic", "worthless"]
    hits = sum(1 for w in crude_flags if w in text.lower())
    return min(1.0, hits * 0.35)


def is_text_nsfw(text: str, toxicity_score: float) -> bool:
    """Flags text as NSFW via explicit-word pattern match OR very high severe toxicity."""
    if _EXPLICIT_TEXT_REGEX.search(text or ""):
        return True
    # Very high toxicity scores often co-occur with graphic/explicit abuse language.
    return toxicity_score >= 0.85


def is_image_nsfw(image_url: str, confidence_threshold: float = 0.60) -> bool:
    """Downloads an image and classifies it as explicit/graphic via NudeNet."""
    model = _load_nsfw_image_model()
    if model is None:
        return False
    try:
        resp = requests.get(image_url, timeout=5)
        resp.raise_for_status()
        tmp_path = os.path.join(BASE_DIR, "_tmp_nsfw_check.jpg")
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        result = model.classify(tmp_path)
        os.remove(tmp_path)
        # NudeClassifier.classify -> {path: {"safe": p_safe, "unsafe": p_unsafe}}
        scores = result.get(tmp_path, {})
        unsafe_score = scores.get("unsafe", 0.0)
        return unsafe_score > confidence_threshold
    except Exception as exc:
        print(f"[main.py] Image NSFW check failed for {image_url}: {exc}")
        return False


def semantic_trigger_match(text: str, user_triggers: List[str], threshold: float = 0.40):
    """
    Embeds `text` and each trigger phrase, returns (matched: bool, best_trigger: str|None,
    best_score: float) using cosine similarity — catches paraphrases, not just keywords.
    """
    if not user_triggers:
        return False, None, 0.0

    model = _load_sentence_model()
    if model is None:
        # Graceful degradation: substring match if the embedding model isn't loaded.
        lowered = (text or "").lower()
        for trig in user_triggers:
            if trig.strip() and trig.strip().lower() in lowered:
                return True, trig, 1.0
        return False, None, 0.0

    try:
        from sentence_transformers import util

        post_vec = model.encode(text, convert_to_tensor=True)
        trigger_vecs = model.encode(user_triggers, convert_to_tensor=True)
        sims = util.cos_sim(post_vec, trigger_vecs)[0]
        best_idx = int(np.argmax(sims.cpu().numpy()))
        best_score = float(sims[best_idx])
        if best_score > threshold:
            return True, user_triggers[best_idx], best_score
        return False, None, best_score
    except Exception as exc:
        print(f"[main.py] Semantic trigger matching failed ({exc}).")
        return False, None, 0.0


def suggest_rephrase(text: str) -> str:
    """
    Rule-based constructive rephrasing suggestion (no commercial LLM API).
    Softens common hostile constructs; this is intentionally simple/template
    based to satisfy the "no commercial LLM doing all the work" constraint.
    """
    replacements = [
        (r"\byou'?re\s+(an?\s+)?idiot\b", "I disagree with your point"),
        (r"\byou\s+are\s+(an?\s+)?idiot\b", "I disagree with your point"),
        (r"\byou'?re\s+(an?\s+)?(moron|stupid|dumb)\b", "I think you're mistaken here"),
        (r"\byou\s+are\s+(an?\s+)?(moron|stupid|dumb)\b", "I think you're mistaken here"),
        (r"\bshut up\b", "please let me finish"),
        (r"\bi hate you\b", "I'm really frustrated right now"),
        (r"\bkill yourself\b", "I strongly disagree with you"),
        (r"\byou\s+are\s+(pathetic|worthless|garbage|trash)\b", "I don't agree with this at all"),
        (r"\bshut\s*up\b", "please stop for a second"),
    ]
    rephrased = text
    for pattern, replacement in replacements:
        rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)

    if rephrased == text:
        # Generic softening fallback: strip repeated punctuation/caps shouting
        rephrased = re.sub(r"!{2,}", "!", rephrased)
        rephrased = re.sub(r"\b[A-Z]{4,}\b", lambda m: m.group(0).title(), rephrased)
        if rephrased == text:
            rephrased = (
                "Consider rewording this more constructively — e.g. explain "
                "why you disagree instead of attacking the other person."
            )
    return rephrased


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class AnalyzePostRequest(BaseModel):
    text: str = ""
    image_urls: List[str] = Field(default_factory=list)
    user_triggers: List[str] = Field(default_factory=list)
    toxicity_threshold: float = 0.5


class AnalyzePostResponse(BaseModel):
    action: str            # "blur" | "show"
    is_toxic: bool
    is_nsfw: bool
    trigger_matched: bool
    matched_trigger: Optional[str]
    reason: str
    toxicity_score: float


class CheckDraftRequest(BaseModel):
    draft_text: str


class CheckDraftResponse(BaseModel):
    is_risky: bool
    toxicity_score: float
    suggestion: Optional[str]


class MoodImpactRequest(BaseModel):
    session_duration_minutes: float
    toxic_blocked_count: int = 0
    nsfw_blocked_count: int = 0
    trigger_blocked_count: int = 0


class MoodImpactResponse(BaseModel):
    stress_saved_units: float
    mood_preservation_score: float
    summary: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "primary_model_loaded": _load_custom_toxic_model() is not None,
        "using_fallback_toxicity": _using_fallback_toxicity,
    }


@app.post("/analyze-post", response_model=AnalyzePostResponse)
def analyze_post(payload: AnalyzePostRequest):
    toxicity_score = score_toxicity(payload.text)
    is_toxic = toxicity_score >= payload.toxicity_threshold

    text_nsfw = is_text_nsfw(payload.text, toxicity_score)
    image_nsfw = any(is_image_nsfw(url) for url in payload.image_urls)
    is_nsfw = text_nsfw or image_nsfw

    trigger_matched, matched_trigger, _ = semantic_trigger_match(
        payload.text, payload.user_triggers
    )

    should_blur = is_toxic or is_nsfw or trigger_matched
    action = "blur" if should_blur else "show"

    if is_toxic and is_nsfw:
        reason = "Toxic and NSFW content detected"
    elif is_toxic:
        reason = "Toxic Content Blocked"
    elif is_nsfw:
        reason = "NSFW Content Blocked"
    elif trigger_matched:
        reason = f"Topic Trigger Matched: '{matched_trigger}'"
    else:
        reason = "No issues detected"

    return AnalyzePostResponse(
        action=action,
        is_toxic=is_toxic,
        is_nsfw=is_nsfw,
        trigger_matched=trigger_matched,
        matched_trigger=matched_trigger,
        reason=reason,
        toxicity_score=round(toxicity_score, 4),
    )


@app.post("/check-draft", response_model=CheckDraftResponse)
def check_draft(payload: CheckDraftRequest):
    score = score_toxicity(payload.draft_text)
    is_risky = score >= 0.45
    suggestion = suggest_rephrase(payload.draft_text) if is_risky else None
    return CheckDraftResponse(
        is_risky=is_risky,
        toxicity_score=round(score, 4),
        suggestion=suggestion,
    )


@app.post("/train-custom-classifier")
def train_custom_classifier():
    """
    Re-runs train_model.py as a subprocess (proves live self-training during
    judging), then reloads the freshly-saved model into memory.
    """
    global _custom_toxic_model, _using_fallback_toxicity

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT_PATH],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Training timed out after 300s."}

    if result.returncode != 0:
        return {
            "status": "error",
            "message": "train_model.py failed.",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # Force reload
    _custom_toxic_model = None
    model = _load_custom_toxic_model()
    _using_fallback_toxicity = model is None

    return {
        "status": "success",
        "message": "Custom toxicity classifier retrained and reloaded.",
        "log": result.stdout.strip().splitlines()[-6:],  # last few lines of training log
    }


@app.post("/calculate-mood-impact", response_model=MoodImpactResponse)
def calculate_mood_impact(payload: MoodImpactRequest):
    n_toxic = payload.toxic_blocked_count
    n_nsfw = payload.nsfw_blocked_count
    n_trigger = payload.trigger_blocked_count
    minutes = max(0.0, payload.session_duration_minutes)

    s_raw = (n_toxic * 2.0) + (n_nsfw * 2.5) + (n_trigger * 1.5)
    density = s_raw / max(1.0, minutes)
    mood_score = max(0.0, min(100.0, 100 - (density * 15) + (s_raw * 2.5)))

    total_blocked = n_toxic + n_nsfw + n_trigger
    if total_blocked == 0:
        summary = (
            f"No harmful content encountered in this {minutes:.0f}-minute session. "
            "Your feed stayed clean."
        )
    else:
        summary = (
            f"Blocked {total_blocked} harmful item(s) "
            f"({n_toxic} toxic, {n_nsfw} NSFW, {n_trigger} trigger matches) "
            f"over {minutes:.0f} minutes. Estimated mental peace preserved: "
            f"{mood_score:.0f}%."
        )

    return MoodImpactResponse(
        stress_saved_units=round(s_raw, 2),
        mood_preservation_score=round(mood_score, 2),
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
