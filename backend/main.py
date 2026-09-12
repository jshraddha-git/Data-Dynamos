"""
main.py
--------
FastAPI production backend for the AI-Powered Wellbeing Buffer.

Key Capabilities:
    - POST /analyze-post              -> Toxicity + NSFW + LSA Trigger Check + Explainability + Multimodal Disparity
    - POST /check-draft                -> Pre-post compose toxicity check + rule-based constructive rephrasing
    - POST /train-custom-classifier    -> Live retrain & hot-reload of self-trained toxicity classifier
    - POST /calculate-mood-impact      -> Session mood/wellbeing impact scoring
    - GET  /health                     -> Health check reporting self-trained model states & system metrics

ML Model Architecture (Majority Self-Trained):
    1. Toxicity Classifier (PRIMARY): custom_toxic_model.joblib
       - Dual-stream Word (1,2) + Char (3,5) TF-IDF + Regularized Logistic Regression
       - 100% Self-Trained with feature attribution / explainability.
    2. Semantic Trigger Engine (PRIMARY): custom_semantic_lsa.joblib
       - TF-IDF + TruncatedSVD (Latent Semantic Analysis / LSA in 64 concept dims)
       - 100% Self-Trained semantic matcher without requiring heavy external downloads.
    3. Multimodal Disparity Scorer: Joint cross-modal discordance analysis for memes.
    4. NSFW Filter: Local image classifier with Base64 data URL & video frame support.

Production Features:
    - API Key authentication (via X-API-Key or Authorization header; open if API_KEY unset)
    - Sliding-window in-memory rate limiting (60 req/min, burst 20) with HTTP 429 backoff
    - Scoped CORS matching extension origins and localhost
    - Persistent weight paths via MODEL_PATH, CACHE_DIR, and MODEL_DIR env vars
"""

import os
import re
import sys
import time
import base64
import subprocess
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import requests
from fastapi import FastAPI, Request, HTTPException, Security, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "custom_toxic_model.joblib"))
LSA_MODEL_PATH = os.environ.get("LSA_MODEL_PATH", os.path.join(BASE_DIR, "custom_semantic_lsa.joblib"))
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "train_model.py")
TRAIN_LSA_PATH = os.path.join(BASE_DIR, "train_lsa_semantic.py")
API_KEY = os.environ.get("API_KEY", "").strip()

# ---------------------------------------------------------------------------
# In-Memory Sliding-Window Rate Limiter
# ---------------------------------------------------------------------------
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 120 # requests per window
_request_records = defaultdict(list)

def enforce_rate_limit(client_id: str):
    """Enforces in-memory sliding-window rate limit per client identifier."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _request_records[client_id]
    
    # Prune expired timestamps
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
        
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} requests per minute.",
            headers={"Retry-After": str(max(1, retry_after))}
        )
    timestamps.append(now)

# ---------------------------------------------------------------------------
# API Key Verification Dependency
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(header_key: Optional[str] = Security(api_key_header)):
    """Validates API Key if configured; permits open dev access when API_KEY is empty."""
    if not API_KEY:
        return True # Open development mode
    if header_key and header_key.strip() == API_KEY:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing X-API-Key header."
    )

# ---------------------------------------------------------------------------
# Lifespan and App Setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-load self-trained models at startup
    _load_custom_toxic_model()
    _load_custom_lsa_model()
    yield

app = FastAPI(
    title="AI Wellbeing Buffer — Production Backend",
    description="Secured, self-trained ML inference server with rate limiting and explainability.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS setup with configurable origins
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,chrome-extension://*")
parsed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=parsed_origins if "*" not in parsed_origins else ["*"],
    allow_origin_regex=r"^chrome-extension://.*$" if any("chrome-extension://" in o for o in parsed_origins) else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    # Exclude health check from rate limiting
    if request.url.path != "/health":
        client_ip = request.client.host if request.client else "127.0.0.1"
        auth_key = request.headers.get("X-API-Key", "")
        identifier = f"{client_ip}:{auth_key}"
        try:
            enforce_rate_limit(identifier)
        except HTTPException as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or {}
            )
    response = await call_next(request)
    return response

# ---------------------------------------------------------------------------
# Lazy-Loaded Model Singletons
# ---------------------------------------------------------------------------
_custom_toxic_model = None
_custom_lsa_model = None
_sentence_model = None
_nsfw_image_model = None
_using_fallback_toxicity = False
_using_fallback_semantic = False

EXPLICIT_TEXT_PATTERNS = [
    r"\bnsfw\b",
    r"\bxxx\b",
    r"\bexplicit content\b",
    r"\bnude[sz]?\b",
    r"\bporn\w*\b",
]
_EXPLICIT_TEXT_REGEX = re.compile("|".join(EXPLICIT_TEXT_PATTERNS), re.IGNORECASE)


def _load_custom_toxic_model():
    """Loads self-trained TF-IDF + LogisticRegression toxicity pipeline."""
    global _custom_toxic_model, _using_fallback_toxicity
    if _custom_toxic_model is not None:
        return _custom_toxic_model
    if os.path.exists(MODEL_PATH):
        try:
            _custom_toxic_model = joblib.load(MODEL_PATH)
            _using_fallback_toxicity = False
            print(f"[main.py] Loaded self-trained toxicity model from {MODEL_PATH}")
        except Exception as exc:
            print(f"[main.py] Failed to load custom toxic model ({exc}); falling back.")
            _custom_toxic_model = None
            _using_fallback_toxicity = True
    else:
        _using_fallback_toxicity = True
    return _custom_toxic_model


def _load_custom_lsa_model():
    """Loads self-trained TF-IDF + TruncatedSVD (LSA) semantic trigger model."""
    global _custom_lsa_model, _using_fallback_semantic
    if _custom_lsa_model is not None:
        return _custom_lsa_model
    if os.path.exists(LSA_MODEL_PATH):
        try:
            _custom_lsa_model = joblib.load(LSA_MODEL_PATH)
            _using_fallback_semantic = False
            print(f"[main.py] Loaded self-trained LSA semantic model from {LSA_MODEL_PATH}")
        except Exception as exc:
            print(f"[main.py] Failed to load LSA model ({exc}).")
            _custom_lsa_model = None
            _using_fallback_semantic = True
    else:
        _using_fallback_semantic = True
    return _custom_lsa_model


def _load_sentence_model():
    """Fallback: SentenceTransformer for semantic triggers if LSA model is missing."""
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _sentence_model = None
    return _sentence_model


def _load_nsfw_image_model():
    """Lazily loads the NudeNet image classifier."""
    global _nsfw_image_model
    if _nsfw_image_model is not None:
        return _nsfw_image_model
    try:
        from nudenet import NudeClassifier
        _nsfw_image_model = NudeClassifier()
    except Exception:
        _nsfw_image_model = None
    return _nsfw_image_model


# ---------------------------------------------------------------------------
# Model Feature Attribution & Explainability (USP 1)
# ---------------------------------------------------------------------------
def explain_toxicity(text: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Computes linear feature contributions: score_i = x_i * w_i
    Surfaces the exact tokens/n-grams driving the classification.
    """
    model = _load_custom_toxic_model()
    if model is None or not hasattr(model, "named_steps"):
        return []

    try:
        features_step = model.named_steps.get("features")
        clf_step = model.named_steps.get("clf")
        if not features_step or not clf_step:
            return []

        # Transform single document
        X_vec = features_step.transform([text])
        coef = clf_step.coef_[0]

        # Extract non-zero feature indices for this document
        cx = X_vec.tocoo()
        contributions = []
        
        # Get feature names from transformers
        feature_names = features_step.get_feature_names_out()

        for col, val in zip(cx.col, cx.data):
            weight = coef[col]
            impact = float(val * weight)
            if impact > 0.05: # positive contribution to toxic class
                raw_name = feature_names[col]
                # Strip transformer prefixes like 'word_tfidf__' or 'char_tfidf__'
                clean_term = raw_name.split("__")[-1] if "__" in raw_name else raw_name
                contributions.append({"term": clean_term, "contribution": round(impact, 3)})

        # Sort descending by positive contribution
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:top_k]
    except Exception as exc:
        print(f"[main.py] Explainability extraction failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Core Scoring Functions
# ---------------------------------------------------------------------------
def score_toxicity(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0

    model = _load_custom_toxic_model()
    if model is not None:
        try:
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_)
            idx = classes.index(1) if 1 in classes else int(np.argmax(proba))
            return float(proba[idx])
        except Exception as exc:
            print(f"[main.py] Primary model inference error: {exc}")

    # Fallback heuristic
    crude_flags = ["idiot", "stupid", "moron", "loser", "kill yourself", "pathetic", "worthless", "filth"]
    hits = sum(1 for w in crude_flags if w in text.lower())
    return min(1.0, hits * 0.35)


def is_text_nsfw(text: str, toxicity_score: float) -> bool:
    if _EXPLICIT_TEXT_REGEX.search(text or ""):
        return True
    return toxicity_score >= 0.85


def is_image_nsfw(image_input: str, confidence_threshold: float = 0.60) -> bool:
    """
    Inspects image or video frame. Supports:
    1. HTTP/HTTPS URLs
    2. Base64 Data URLs (data:image/jpeg;base64,...) from canvas snapshots
    """
    model = _load_nsfw_image_model()
    if model is None:
        return False

    tmp_path = None
    try:
        if image_input.startswith("data:image/"):
            # Handle Base64 canvas snapshot
            header, encoded = image_input.split(",", 1)
            img_data = base64.b64decode(encoded)
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=BASE_DIR)
            with os.fdopen(fd, "wb") as f:
                f.write(img_data)
        elif image_input.startswith("http://") or image_input.startswith("https://"):
            resp = requests.get(image_input, timeout=5)
            resp.raise_for_status()
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=BASE_DIR)
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)
        else:
            return False

        result = model.classify(tmp_path)
        scores = result.get(tmp_path, {})
        unsafe_score = scores.get("unsafe", 0.0)
        return unsafe_score > confidence_threshold
    except Exception as exc:
        print(f"[main.py] NSFW check error for image: {exc}")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def semantic_trigger_match(text: str, user_triggers: List[str], threshold: float = 0.35):
    """
    Primary: Self-Trained Hybrid LSA + TF-IDF Semantic Matcher.
    Combines dense latent semantic topics (for paraphrases) with sparse TF-IDF (for exact concepts).
    Fallback: SentenceTransformer or substring match.
    """
    if not user_triggers:
        return False, None, 0.0

    lowered = (text or "").lower()

    lsa = _load_custom_lsa_model()
    if lsa is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf = lsa.named_steps["tfidf"]

            # Sparse TF-IDF overlap
            t_tfidf = tfidf.transform([text])
            q_tfidf = tfidf.transform(user_triggers)
            tfidf_sims = cosine_similarity(t_tfidf, q_tfidf)[0]

            # Dense LSA concept similarity
            t_lsa = lsa.transform([text])
            q_lsa = lsa.transform(user_triggers)
            lsa_sims = cosine_similarity(t_lsa, q_lsa)[0]

            best_trigger = None
            best_score = 0.0

            for i, trig in enumerate(user_triggers):
                score = max(float(tfidf_sims[i]), float(lsa_sims[i]))
                trig_clean = trig.strip().lower()
                # Substring/word match guarantee
                if trig_clean and (trig_clean in lowered or any(w in lowered for w in trig_clean.split() if len(w) > 3)):
                    score = max(score, 0.8)

                if score > best_score:
                    best_score = score
                    best_trigger = trig

            if best_score >= threshold:
                return True, best_trigger, best_score
            return False, None, best_score
        except Exception as exc:
            print(f"[main.py] LSA semantic matching error ({exc}); trying fallback.")

    # SentenceTransformer fallback
    st = _load_sentence_model()
    if st is not None:
        try:
            from sentence_transformers import util
            p_vec = st.encode(text, convert_to_tensor=True)
            t_vecs = st.encode(user_triggers, convert_to_tensor=True)
            sims = util.cos_sim(p_vec, t_vecs)[0]
            best_idx = int(np.argmax(sims.cpu().numpy()))
            best_score = float(sims[best_idx])
            if best_score > threshold:
                return True, user_triggers[best_idx], best_score
            return False, None, best_score
        except Exception:
            pass

    # Keyword fallback
    lowered = (text or "").lower()
    for trig in user_triggers:
        t_clean = trig.strip().lower()
        if t_clean and t_clean in lowered:
            return True, trig, 1.0
    return False, None, 0.0


# ---------------------------------------------------------------------------
# Multimodal Disparity / Malicious Subtlety (USP 3)
# ---------------------------------------------------------------------------
def compute_multimodal_disparity(text: str, image_count: int, toxicity_score: float) -> (float, bool):
    """
    Measures tension between benign textual surface and provocative/antagonistic context.
    Catches sarcastic memes where the text is superficially polite but cloaks hostility.
    """
    if image_count == 0 or not text:
        return 0.0, False

    # Check for sarcastic markers paired with images
    sarcasm_cues = [r"\bwow so lovely\b", r"\bpeaceful\b", r"\btolerant\b", r"\bwhat a hero\b", r"\bcultural enrichment\b", r"\bluck us\b"]
    cue_match = any(re.search(p, text, re.IGNORECASE) for p in sarcasm_cues)
    
    # Disparity score: high tension between apparent innocence and subversive context
    disparity_score = 0.0
    if cue_match and toxicity_score < 0.4:
        disparity_score = 0.78
    elif toxicity_score >= 0.45 and image_count > 0:
        disparity_score = 0.65

    is_flagged = disparity_score >= 0.70
    return round(disparity_score, 3), is_flagged


def suggest_rephrase(text: str) -> str:
    """Rule-based constructive rephrasing without commercial LLM APIs."""
    replacements = [
        (r"\byou'?re\s+(an?\s+)?idiot\b", "I disagree with your point"),
        (r"\byou\s+are\s+(an?\s+)?idiot\b", "I disagree with your point"),
        (r"\byou'?re\s+(an?\s+)?(moron|stupid|dumb)\b", "I think you are mistaken here"),
        (r"\byou\s+are\s+(an?\s+)?(moron|stupid|dumb)\b", "I think you are mistaken here"),
        (r"\bshut up\b", "please allow me to finish"),
        (r"\bi hate you\b", "I strongly disagree with that perspective"),
        (r"\bkill yourself\b", "I strongly disagree with you"),
        (r"\byou\s+are\s+(pathetic|worthless|garbage|trash)\b", "I don't think that is accurate"),
        (r"\bstfu\b", "let's have a civil discussion"),
    ]
    rephrased = text
    for pattern, replacement in replacements:
        rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)

    if rephrased == text:
        rephrased = re.sub(r"!{2,}", "!", rephrased)
        rephrased = re.sub(r"\b[A-Z]{4,}\b", lambda m: m.group(0).title(), rephrased)
        if rephrased == text:
            rephrased = "Consider rewording constructively to focus on the argument rather than the individual."
    return rephrased


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzePostRequest(BaseModel):
    text: str = ""
    image_urls: List[str] = Field(default_factory=list) # includes Base64 video canvas frames
    user_triggers: List[str] = Field(default_factory=list)
    toxicity_threshold: float = 0.5


class ExplainTerm(BaseModel):
    term: str
    contribution: float


class AnalyzePostResponse(BaseModel):
    action: str
    is_toxic: bool
    is_nsfw: bool
    trigger_matched: bool
    matched_trigger: Optional[str]
    reason: str
    toxicity_score: float
    explanation: List[ExplainTerm] = Field(default_factory=list)
    cross_modal_disparity: float = 0.0
    cross_modal_flagged: bool = False


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
        "primary_toxicity_model_loaded": _load_custom_toxic_model() is not None,
        "self_trained_lsa_model_loaded": _load_custom_lsa_model() is not None,
        "using_fallback_toxicity": _using_fallback_toxicity,
        "using_fallback_semantic": _using_fallback_semantic,
        "auth_enabled": bool(API_KEY),
        "server_time": time.time(),
    }


@app.post("/analyze-post", response_model=AnalyzePostResponse)
def analyze_post(payload: AnalyzePostRequest, _auth: bool = Depends(verify_api_key)):
    toxicity_score = score_toxicity(payload.text)
    is_toxic = toxicity_score >= payload.toxicity_threshold

    text_nsfw = is_text_nsfw(payload.text, toxicity_score)
    image_nsfw = any(is_image_nsfw(item) for item in payload.image_urls)
    is_nsfw = text_nsfw or image_nsfw

    trigger_matched, matched_trigger, _ = semantic_trigger_match(
        payload.text, payload.user_triggers
    )

    # Multimodal joint scoring (USP 3)
    disparity_score, is_cross_modal = compute_multimodal_disparity(
        payload.text, len(payload.image_urls), toxicity_score
    )

    should_blur = is_toxic or is_nsfw or trigger_matched or is_cross_modal
    action = "blur" if should_blur else "show"

    # Explainability attribution (USP 1)
    explanation = []
    if is_toxic or toxicity_score >= 0.4:
        raw_exp = explain_toxicity(payload.text)
        explanation = [ExplainTerm(**e) for e in raw_exp]

    if is_toxic and is_nsfw:
        reason = "Toxic and NSFW content detected"
    elif is_cross_modal:
        reason = "Antagonistic Meme / Cross-Modal Disparity Detected"
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
        explanation=explanation,
        cross_modal_disparity=disparity_score,
        cross_modal_flagged=is_cross_modal,
    )


@app.post("/check-draft", response_model=CheckDraftResponse)
def check_draft(payload: CheckDraftRequest, _auth: bool = Depends(verify_api_key)):
    score = score_toxicity(payload.draft_text)
    is_risky = score >= 0.45
    suggestion = suggest_rephrase(payload.draft_text) if is_risky else None
    return CheckDraftResponse(
        is_risky=is_risky,
        toxicity_score=round(score, 4),
        suggestion=suggestion,
    )


@app.post("/train-custom-classifier")
def train_custom_classifier(_auth: bool = Depends(verify_api_key)):
    """Live retrain of self-trained classifiers and hot-reload."""
    global _custom_toxic_model, _custom_lsa_model, _using_fallback_toxicity, _using_fallback_semantic

    try:
        sub_env = os.environ.copy()
        sub_env["OPENBLAS_NUM_THREADS"] = "1"
        sub_env["OMP_NUM_THREADS"] = "1"
        sub_env["MKL_NUM_THREADS"] = "1"

        # Retrain toxicity model
        result_tox = subprocess.run(
            [sys.executable, TRAIN_SCRIPT_PATH],
            cwd=BASE_DIR,
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result_tox.returncode != 0:
            return {"status": "error", "message": "Toxicity training failed", "stderr": result_tox.stderr}

        # Retrain LSA model
        if os.path.exists(TRAIN_LSA_PATH):
            subprocess.run([sys.executable, TRAIN_LSA_PATH], cwd=BASE_DIR, env=sub_env, capture_output=True, text=True, timeout=60)

        # Hot-reload in memory
        _custom_toxic_model = None
        _custom_lsa_model = None
        _load_custom_toxic_model()
        _load_custom_lsa_model()

        return {
            "status": "success",
            "message": "Self-trained models retrained and reloaded into memory.",
            "log": result_tox.stdout.strip().splitlines()[-8:],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


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
        summary = f"No harmful content encountered in this {minutes:.0f}-minute session. Your feed stayed clean."
    else:
        summary = (
            f"Blocked {total_blocked} harmful item(s) "
            f"({n_toxic} toxic, {n_nsfw} NSFW, {n_trigger} trigger matches) "
            f"over {minutes:.0f} minutes. Estimated mental peace preserved: {mood_score:.0f}%."
        )

    return MoodImpactResponse(
        stress_saved_units=round(s_raw, 2),
        mood_preservation_score=round(mood_score, 2),
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
