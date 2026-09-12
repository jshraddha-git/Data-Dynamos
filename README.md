# AI Wellbeing Buffer

> **An AI-powered moderation layer that filters toxic, NSFW, and trigger content before it renders on social feeds, featuring genuine self-trained ML models, explainability attribution, adaptive sensitivity learning, and production container deployment.**

Built for **Can You Hack It** under the strict constraint: **no commercial LLM APIs (OpenAI / Anthropic / Gemini)** — all core ML capabilities run on models trained by our team.

---

## 🌟 Key Capabilities & Novelty (USPs)

1. **Majority Self-Trained ML Architecture**:
   - **Toxicity Classifier (100% Self-Trained)**: Dual-stream Word (1,2) + Character (3,5) TF-IDF feature union with L2-regularized Logistic Regression. Catches leetspeak and obfuscated slurs without tripping on benign slang ("killed that solo", "sick kickflip").
   - **Semantic Trigger Engine (100% Self-Trained)**: Latent Semantic Analysis (LSA) projecting TF-IDF into 64 dense concept dimensions via Singular Value Decomposition (TruncatedSVD). Matches user-defined trigger concepts (e.g. "layoffs", "spoilers", "crypto scams") without requiring heavy deep learning weights.
2. **Model Explainability on Flags (Attribution)**:
   - Eliminates "black box" filtering. The overlay includes an interactive *"Why was this flagged?"* chip that surfaces the exact n-grams and mathematical weights that triggered the classification (e.g. `pathetic (+0.25)`, `moron (+0.20)`).
3. **Adaptive Personal Sensitivity (Online Feedback Loop)**:
   - Learns from implicit and explicit user interaction (unhiding / rehiding posts). Borderline content tolerance automatically adjusts per-user locally without leaking private browsing history.
4. **Multimodal Joint Scoring ("Malicious Subtlety")**:
   - Detects discordant memes where text appears superficially polite but visual context creates antagonistic tension.
5. **Multi-Platform Adapter Architecture**:
   - Dedicated DOM adapters for **Twitter/X, Reddit, Instagram, Facebook, TikTok, LinkedIn, YouTube**, plus an automatic generic fallback for blogs and forums.
   - Extracts both static images and active HTML5 `<video>` canvas frame snapshots for media moderation.

---

## 📊 Honest Model Evaluation & Before/After Metrics

The original prototype achieved an artificial 100% accuracy on a 500-row synthetic dataset by memorizing templated phrasing. When evaluated against real-world conversational edge cases, it suffered from severe false positives (flagging benign slang and civil disagreements).

We broadened and diversified the corpus to eliminate data leakage and re-trained with a dual-stream n-gram architecture:

| Metric | Original Baseline (Synthetic Only) | Expanded Self-Trained Model |
| :--- | :---: | :---: |
| **Training Corpus** | 500 templated rows | 617 unique, non-leaking diverse rows |
| **Held-out Test Split** | 100 rows | 124 held-out rows (stratified) |
| **Held-out Accuracy** | 100% *(overfit)* | **99.19%** *(realistic generalization)* |
| **Precision** | 1.0000 | **1.0000** |
| **Recall** | 1.0000 | **0.9836** (1 false negative) |
| **F1-Score (Binary)** | 1.0000 | **0.9917** |
| **F1-Score (Macro)** | 1.0000 | **0.9919** |
| **False Positives on Benign Slang** | **51 / 311 (16.4% false alarms)** | **0 / 311 (0% false alarms)** |

### Confusion Matrix on Held-Out Test Split (124 samples):
```
                       Predicted Clean    Predicted Toxic
Actual Clean (63)            63                  0
Actual Toxic (61)             1                 60
```
*Note: The model correctly allowed colloquial phrases like "you absolutely killed that guitar solo" and "that new album is sick" while catching real abuse and leetspeak attacks.*

---

## 🏗 System Architecture

```
                                  CHROME EXTENSION (MV3)
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                        │
  │   Content Script ──────► Platform Adapter Layer (Twitter, Reddit, IG, TikTok, YT...)   │
  │         │                 │                                                            │
  │         │                 ├─► Text Extraction (captions, comments, tweets)             │
  │         │                 └─► Media Extraction (images, posters, video canvas frames)  │
  │         ▼                                                                              │
  │   Service Worker (background.js) ◄─── Centralized Cross-Tab Queue & Auth Manager       │
  └─────────┬──────────────────────────────────────────────────────────────────────────────┘
            │ HTTPS / POST (with X-API-Key)
            ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                               FASTAPI BACKEND CONTAINER                                │
  │                                                                                        │
  │   [Security Layer] ──► API Key Verification & In-Memory Sliding-Window Rate Limiter    │
  │                                                                                        │
  │   [ML Engine 1: Toxicity]      TF-IDF Word (1,2) + Char (3,5) + Logistic Regression     │
  │   [ML Engine 2: Triggers]      Latent Semantic Analysis (TF-IDF + TruncatedSVD 64d)     │
  │   [ML Engine 3: Multimodal]    Cross-Modal Cosine Disparity Scorer                      │
  │   [ML Engine 4: Media NSFW]    Video Frame & Image Classifier                            │
  │   [Explainability Engine]      Linear Feature Attribution (Token Weights)              │
  │                                                                                        │
  │   [Live Retrain Endpoint]      /train-custom-classifier (hot-reloads into memory)      │
  │   [Storage Volume]             /app/models (joblib) & /app/cache (persistent weights)  │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Production Deployment Guide

### Option 1: Local Docker / Docker Compose

To run the containerized backend locally with full production parity:

```bash
# Clone and enter directory
cd ai-wellbeing-layer

# Build and launch with volume persistence
docker compose up --build -d

# Verify health check
curl http://localhost:8000/health
```

### Option 2: Deploy to Fly.io (with Persistent Volume)

Fly.io provides automated HTTPS and persistent NVMe volume mounting:

```bash
# 1. Install Fly CLI and login
fly auth login

# 2. Launch using fly.toml
fly launch --no-deploy

# 3. Create persistent storage volume for model weights and cache
fly volumes create wellbeing_storage --size 5 --region iad

# 4. Set secrets
fly secrets set API_KEY="your-production-secret-key"

# 5. Deploy
fly deploy
```

### Option 3: Deploy to Render

1. Fork or push this repository to GitHub.
2. In Render Dashboard, click **New > Blueprint**.
3. Select this repo — Render will automatically detect `render.yaml`, provision a web service with a 10GB persistent disk at `/app/cache`, and assign a public HTTPS URL.

### Option 4: Cloud VM (Ubuntu VPS + Caddy HTTPS Reverse Proxy)

```bash
# 1. Install Caddy for automatic Let's Encrypt TLS
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLF 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLF 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy

# 2. Configure /etc/caddy/Caddyfile
your-domain.com {
    reverse_proxy localhost:8000
}

# 3. Run the container
docker run -d --restart unless-stopped -p 8000:8000 \
  -e API_KEY="prod-secret-key" \
  -v wellbeing_cache:/app/cache \
  wellbeing_backend
```

---

## 🧩 Chrome Extension Setup

1. Open Google Chrome and navigate to `chrome://extensions`.
2. Toggle on **Developer mode** in the top right.
3. Click **Load unpacked** and select the `ai-wellbeing-layer/extension/` directory.
4. Click the **AI Wellbeing Buffer** extension icon in your toolbar.
5. In **⚙ Server Connection**:
   - For local development: leave as `http://localhost:8000`.
   - For production: enter your hosted URL (e.g. `https://your-app.fly.dev`) and your `API_KEY`.
6. Navigate to Twitter/X, Reddit, Instagram, Facebook, TikTok, LinkedIn, or YouTube and experience the protective layer.

---

## 🎯 Demo Walkthrough (For Judges)

1. **Verify Self-Trained Capabilities (`/health`)**:
   - Check `GET /health` to show both `primary_toxicity_model_loaded: true` and `self_trained_lsa_model_loaded: true`.
2. **Browse Feed with Explainability**:
   - Scroll a live social feed. When a toxic post is blurred, click *"ℹ Why was this flagged?"* on the overlay to show exact contributing tokens.
3. **Adaptive Personal Sensitivity**:
   - Click *"Unhide Content"* on borderline comments. Open the popup to observe the live *"Shift: +0.04"* indicator learning your personal tolerance.
4. **LSA Semantic Concept Matching**:
   - Add trigger topic `"layoffs"` in popup. Note how it flags paraphrased corporate downsizing announcements even when the exact keyword is omitted.
5. **Live Model Retraining**:
   - Trigger `POST /train-custom-classifier` live in terminal or curl. Watch the system retrain on the spot and hot-reload weights without downtime.
