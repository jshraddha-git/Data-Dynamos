"""
main.py
--------
FastAPI backend for the AI-Powered Wellbeing Buffer.

Every model here is trained from scratch by this repo's own training
scripts — nothing pretrained, nothing fetched from a commercial API:

  - Toxicity classifier   -> train_model.py            -> custom_toxic_model.joblib
        TF-IDF + LogisticRegression, trained on data/toxic_sample.csv.
  - Image NSFW classifier -> train_nsfw_model.py        -> nsfw_image_model.joblib
        LogisticRegression over hand-engineered pixel/color features
        (image_features.py), trained on procedurally generated synthetic
        images (no pretrained CNN, no NudeNet).
  - Semantic trigger match-> train_trigger_embedder.py  -> trigger_embedder.joblib
        TF-IDF + Truncated SVD (LSA), trained on our own text corpus
        (no sentence-transformers, no pretrained embeddings).
  - Text NSFW             -> severe-toxicity signal (from the toxicity
        classifier above) + explicit-word pattern matching. No extra model.
  - Draft rephrasing      -> deterministic rule-based rewriter. No model,
        no LLM call of any kind.
  - Mood impact            -> closed-form formula over session stats.

If any .joblib file is missing at startup, main.py trains it itself on the
spot by calling straight into that model's train_and_save() — the fallback
is "train it now", never "load someone else's pretrained weights".

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import re
import io
import logging
import subprocess
import sys
from typing import List, Optional

import joblib
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from image_features import extract_image_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOXIC_MODEL_PATH = os.path.join(BASE_DIR, "custom_toxic_model.joblib")
NSFW_MODEL_PATH = os.path.join(BASE_DIR, "nsfw_image_model.joblib")
TRIGGER_MODEL_PATH = os.path.join(BASE_DIR, "trigger_embedder.joblib")
TRAIN_TOXIC_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Wellbeing Buffer API", version="2.0.0")

# The Chrome extension calls this API from content-script/popup contexts on
# arbitrary sites, so we allow broadly here. In production, lock this down
# to specific extension IDs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model handles — all three are self-trained sklearn artifacts
# ---------------------------------------------------------------------------
toxic_model = None       # sklearn Pipeline: TfidfVectorizer + LogisticRegression
nsfw_model = None        # sklearn LogisticRegression over image_features.py vectors
trigger_embedder = None  # sklearn Pipeline: TfidfVectorizer + TruncatedSVD (LSA)

# Explicit-word pattern list used for text-based NSFW flagging. Kept
# intentionally coarse; this is a pattern-match layer that runs alongside
# (not instead of) the toxicity-derived severe-toxicity signal.
EXPLICIT_TEXT_PATTERNS = [
    r"\bnsfw\b",
    r"\bxxx\b",
    r"\bporn\w*\b",
    r"\bnud(e|ity)\w*\b",
    r"\bexplicit content\b",
    r"\bonlyfans\b",
    r"\bsex\s?tape\b",
]
EXPLICIT_TEXT_REGEX = re.compile("|".join(EXPLICIT_TEXT_PATTERNS), re.IGNORECASE)

NSFW_IMAGE_CONFIDENCE_THRESHOLD = 0.60
SEMANTIC_TRIGGER_THRESHOLD = 0.40
DRAFT_RISK_THRESHOLD = 0.45
SEVERE_TOXICITY_THRESHOLD = 0.75  # stricter than the general toxicity flag


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AnalyzePostRequest(BaseModel):
    text: str = ""
    image_urls: List[str] = Field(default_factory=list)
    user_triggers: List[str] = Field(default_factory=list)
    toxicity_threshold: float = 0.5


class AnalyzePostResponse(BaseModel):
    action: str  # "blur" | "show"
    is_toxic: bool
    is_nsfw: bool
    trigger_matched: bool
    matched_trigger: Optional[str] = None
    reason: str
    toxicity_score: float


class CheckDraftRequest(BaseModel):
    draft_text: str


class CheckDraftResponse(BaseModel):
    is_risky: bool
    toxicity_score: float
    suggestion: Optional[str] = None


class MoodImpactRequest(BaseModel):
    session_duration_minutes: float
    toxic_blocked_count: int
    nsfw_blocked_count: int
    trigger_blocked_count: int


class MoodImpactResponse(BaseModel):
    stress_saved_units: float
    mood_preservation_score: float
    summary: str


class TrainResponse(BaseModel):
    success: bool
    message: str
    n_samples: Optional[int] = None
    holdout_accuracy: Optional[float] = None


# ---------------------------------------------------------------------------
# Startup: load each self-trained model, training it on the spot if missing
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models_on_startup():
    global toxic_model, nsfw_model, trigger_embedder

    # 1) Toxicity classifier
    if not os.path.exists(TOXIC_MODEL_PATH):
        logger.info("custom_toxic_model.joblib not found — training it now.")
        from train_model import train_and_save as train_toxic

        train_toxic()
    toxic_model = joblib.load(TOXIC_MODEL_PATH)
    logger.info("Loaded self-trained toxicity classifier.")

    # 2) Image NSFW classifier
    if not os.path.exists(NSFW_MODEL_PATH):
        logger.info("nsfw_image_model.joblib not found — training it now.")
        from train_nsfw_model import train_and_save as train_nsfw

        train_nsfw()
    nsfw_model = joblib.load(NSFW_MODEL_PATH)
    logger.info("Loaded self-trained NSFW image classifier.")

    # 3) Semantic trigger embedder
    if not os.path.exists(TRIGGER_MODEL_PATH):
        logger.info("trigger_embedder.joblib not found — training it now.")
        from train_trigger_embedder import train_and_save as train_embedder

        train_embedder()
    trigger_embedder = joblib.load(TRIGGER_MODEL_PATH)
    logger.info("Loaded self-trained semantic trigger embedder (TF-IDF + LSA).")

    logger.info("All models loaded. No pretrained weights or external APIs were used.")


# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------
def score_text_toxicity(text: str) -> float:
    """Toxicity probability in [0, 1] from the self-trained TF-IDF + LogisticRegression pipeline."""
    if not text or not text.strip():
        return 0.0

    proba = toxic_model.predict_proba([text])[0]
    classes = list(toxic_model.named_steps["clf"].classes_)
    toxic_idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
    return float(proba[toxic_idx])


def text_contains_explicit_pattern(text: str) -> bool:
    return bool(EXPLICIT_TEXT_REGEX.search(text or ""))


def is_text_nsfw(text: str, toxicity_score: float) -> bool:
    """
    Text NSFW = severe toxicity signal OR explicit-word pattern match.
    We don't have a separate "severity" head, so we reuse the same
    self-trained toxicity score at a stricter threshold as the severity proxy.
    """
    return toxicity_score >= SEVERE_TOXICITY_THRESHOLD or text_contains_explicit_pattern(text)


def check_images_nsfw(image_urls: List[str]) -> bool:
    """
    Downloads each image, extracts the hand-engineered feature vector
    (image_features.py), and runs it through the self-trained
    LogisticRegression NSFW classifier. Returns True if ANY image is
    classified explicit above NSFW_IMAGE_CONFIDENCE_THRESHOLD. Network/
    decode failures on individual images are logged and skipped rather
    than failing the whole request.
    """
    if not image_urls:
        return False

    for url in image_urls:
        try:
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()

            img = Image.open(io.BytesIO(resp.content))
            img.load()  # force decode / validate it's a real image

            features = extract_image_features(img).reshape(1, -1)
            proba = nsfw_model.predict_proba(features)[0]
            classes = list(nsfw_model.classes_)
            explicit_idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
            explicit_confidence = float(proba[explicit_idx])

            if explicit_confidence > NSFW_IMAGE_CONFIDENCE_THRESHOLD:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Skipping image %s during NSFW check (%s).", url, exc)
            continue

    return False


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Push text through the self-trained TF-IDF + TruncatedSVD (LSA) pipeline
    and L2-normalize, so a dot product gives cosine similarity.
    """
    vecs = trigger_embedder.transform(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vecs / norms


def find_matching_trigger(text: str, user_triggers: List[str]) -> Optional[str]:
    """
    Computes cosine similarity, in our self-trained LSA space, between the
    post's embedding and each user trigger phrase's embedding. Returns the
    best-matching trigger if it exceeds SEMANTIC_TRIGGER_THRESHOLD, else None.
    """
    if not text or not text.strip() or not user_triggers:
        return None

    post_vec = embed_texts([text])[0]
    trigger_vecs = embed_texts(user_triggers)

    similarities = trigger_vecs @ post_vec
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score > SEMANTIC_TRIGGER_THRESHOLD:
        return user_triggers[best_idx]
    return None


def generate_rephrase_suggestion(draft_text: str) -> str:
    """
    Produces a constructive rephrasing suggestion WITHOUT calling any model
    or API. This is a deterministic, rule-based rewriter: it softens a small
    dictionary of common hostile phrases and wraps the result in an "I"
    statement, a well-established de-escalation pattern. It's not as fluent
    as an LLM rewrite, but it keeps this feature 100% local and API-free.
    """
    softeners = {
        r"\byou'?re? (an? )?idiot\b": "I disagree with you",
        r"\bshut up\b": "please let me finish",
        r"\bstupid\b": "mistaken",
        r"\bhate\b": "strongly dislike",
        r"\bworthless\b": "not adding value here",
        r"\bpathetic\b": "disappointing",
        r"\bmoron\b": "person who sees it differently",
        r"\bloser\b": "person",
        r"\bkill yourself\b": "please reconsider this",
        r"\bdisgust(s|ing)?\b": "concern(s)",
        r"\bgarbage\b": "not great",
        r"\btrash\b": "not great",
    }

    rewritten = draft_text
    changed = False
    for pattern, replacement in softeners.items():
        new_text, n = re.subn(pattern, replacement, rewritten, flags=re.IGNORECASE)
        if n > 0:
            changed = True
            rewritten = new_text

    if not changed:
        rewritten = (
            "Consider leading with how you feel rather than an accusation, e.g.: "
            f"\"I see this differently than you do because...\" "
            f"(original draft: \"{draft_text.strip()}\")"
        )
    else:
        rewritten = rewritten.strip().capitalize()

    return rewritten


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "models": {
            "toxicity_classifier": "self-trained (TF-IDF + LogisticRegression)",
            "nsfw_image_classifier": "self-trained (hand-engineered features + LogisticRegression)",
            "trigger_embedder": "self-trained (TF-IDF + TruncatedSVD / LSA)",
        },
    }


@app.post("/analyze-post", response_model=AnalyzePostResponse)
def analyze_post(payload: AnalyzePostRequest):
    toxicity_score = score_text_toxicity(payload.text)
    is_toxic = toxicity_score >= payload.toxicity_threshold

    is_nsfw = is_text_nsfw(payload.text, toxicity_score)
    if not is_nsfw and payload.image_urls:
        is_nsfw = check_images_nsfw(payload.image_urls)

    matched_trigger = find_matching_trigger(payload.text, payload.user_triggers)
    trigger_matched = matched_trigger is not None

    if is_toxic:
        reason = f"Toxicity score {toxicity_score:.2f} met/exceeded threshold {payload.toxicity_threshold:.2f}."
    elif is_nsfw:
        reason = "Content flagged as NSFW (explicit text pattern, severe toxicity, or image classifier)."
    elif trigger_matched:
        reason = f"Semantic similarity to trigger topic '{matched_trigger}' exceeded {SEMANTIC_TRIGGER_THRESHOLD}."
    else:
        reason = "No toxicity, NSFW, or trigger-topic signals detected."

    action = "blur" if (is_toxic or is_nsfw or trigger_matched) else "show"

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
    score = score_text_toxicity(payload.draft_text)
    is_risky = score >= DRAFT_RISK_THRESHOLD

    suggestion = generate_rephrase_suggestion(payload.draft_text) if is_risky else None

    return CheckDraftResponse(
        is_risky=is_risky,
        toxicity_score=round(score, 4),
        suggestion=suggestion,
    )


@app.post("/train-custom-classifier", response_model=TrainResponse)
def train_custom_classifier():
    """
    Programmatically re-runs train_model.py as a subprocess (so training
    happens in a clean process), then hot-reloads the resulting joblib file
    into memory so subsequent /analyze-post and /check-draft calls
    immediately use the freshly retrained model. Useful for demoing live
    self-training to judges.
    """
    global toxic_model

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_TOXIC_SCRIPT],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return TrainResponse(success=False, message="Training timed out after 120s.")

    if result.returncode != 0:
        logger.error("train_model.py failed: %s", result.stderr)
        return TrainResponse(success=False, message=f"Training failed: {result.stderr[-500:]}")

    try:
        toxic_model = joblib.load(TOXIC_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        return TrainResponse(success=False, message=f"Trained but failed to reload model: {exc}")

    accuracy = None
    n_samples = None
    for line in result.stdout.splitlines():
        if "Holdout accuracy" in line:
            try:
                accuracy = float(line.strip().split(":")[-1])
            except ValueError:
                pass
        if "Trained on" in line:
            try:
                n_samples = int(line.split("Trained on")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass

    return TrainResponse(
        success=True,
        message="Custom classifier retrained and hot-reloaded successfully.",
        n_samples=n_samples,
        holdout_accuracy=accuracy,
    )


@app.post("/calculate-mood-impact", response_model=MoodImpactResponse)
def calculate_mood_impact(payload: MoodImpactRequest):
    """
    Deterministic, explainable formula (no ML model needed) converting raw
    blocked-content counts + session length into a "Mood Preservation Score".

    S_raw   = (toxic * 2.0) + (nsfw * 2.5) + (trigger * 1.5)
    D       = S_raw / max(1, minutes)
    Score % = clamp(0, 100, 100 - (D * 15) + (S_raw * 2.5))
    """
    s_raw = (
        (payload.toxic_blocked_count * 2.0)
        + (payload.nsfw_blocked_count * 2.5)
        + (payload.trigger_blocked_count * 1.5)
    )

    minutes = max(1.0, payload.session_duration_minutes)
    density = s_raw / minutes

    raw_score = 100 - (density * 15) + (s_raw * 2.5)
    mood_score = max(0.0, min(100.0, raw_score))

    total_blocked = (
        payload.toxic_blocked_count
        + payload.nsfw_blocked_count
        + payload.trigger_blocked_count
    )

    summary = (
        f"Blocked {total_blocked} harmful item(s) over "
        f"{payload.session_duration_minutes:.0f} min — "
        f"{mood_score:.0f}% mental peace preserved."
    )

    return MoodImpactResponse(
        stress_saved_units=round(s_raw, 2),
        mood_preservation_score=round(mood_score, 2),
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
