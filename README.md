# AI Wellbeing Buffer

An AI-powered layer that sits between an online feed and the reader — filtering
toxic content, NSFW media, and personal trigger topics **before** they render,
and showing the reader what an hour of scrolling did to their mood.

Built for **Can You Hack It**. No commercial LLM APIs (OpenAI/Anthropic/Gemini)
are used — all models are self-trained or self-hosted open-source models.

---

## Architecture

```
ai-wellbeing-layer/
├── backend/                  # FastAPI server + local ML models
│   ├── data/toxic_sample.csv     # 500-row labeled training set
│   ├── train_model.py            # trains TF-IDF + LogisticRegression classifier
│   ├── main.py                   # API: /analyze-post, /check-draft, /train-custom-classifier, /calculate-mood-impact
│   └── requirements.txt
└── extension/                # Manifest V3 Chrome extension
    ├── manifest.json
    ├── content.js             # scans feed, applies blur/overlays, pre-post check
    ├── styles.css              # cyberpunk dark theme
    ├── popup.html               # settings + mood dashboard
    └── popup.js
```

**Model precedence (toxicity):** the API tries the team's self-trained
`custom_toxic_model.joblib` (TF-IDF + LogisticRegression) first. If that file
doesn't exist yet, it falls back to `Detoxify('original-small')` so the API
never breaks — but the primary model is what should be running for the demo.

---

## Setup

### 1. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Train the self-owned classifier (creates custom_toxic_model.joblib)
python train_model.py

# Start the API
python main.py
# or: uvicorn main:app --reload --port 8000
```

The API runs at `http://localhost:8000`. Check `http://localhost:8000/health`
to confirm the primary model loaded.

> Note: `sentence-transformers`, `detoxify`, and `nudenet` download open-source
> model weights on first use (internet required once). Everything after that
> runs fully locally.

### 2. Chrome Extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked** → select the `extension/` folder
4. Pin the extension, open a feed (Twitter/X, Reddit, etc.), and scroll

Make sure the backend is running on `localhost:8000` first — the extension
calls it directly from the content script and popup.

---

## Using It

- **Sensitivity slider** (popup): sets the toxicity threshold sent with every
  `/analyze-post` call.
- **Topics to Avoid**: type any topic in plain language (e.g. "spoilers",
  "layoffs") — matched by semantic similarity, not exact keywords.
- **NSFW image filtering** / **Pre-post draft check**: toggle independently.
- **This Session**: live counts of blocked items + a "Mental Peace Preserved"
  score, computed by `/calculate-mood-impact` from this session's block counts
  and elapsed time.
- **Live retraining demo**: `curl -X POST http://localhost:8000/train-custom-classifier`
  re-runs `train_model.py` and hot-reloads the model — useful to show judges
  the model is genuinely self-trained, not a static artifact.

---

## Demo Flow (suggested)

1. Show `/health` returning `"primary_model_loaded": true` — the classifier is
   yours, not a wrapped commercial API.
2. Scroll a feed with the extension on — show blur + liftable badges for
   toxic/NSFW/trigger-matched posts.
3. Type a hostile draft comment — show the inline rephrase suggestion.
4. Add a plain-language trigger topic, show it catching a paraphrased post a
   keyword filter would miss.
5. Open the popup's Mood dashboard — show the live wellbeing score.
6. Call `/train-custom-classifier` live to prove the model is retrainable
   on the spot.
