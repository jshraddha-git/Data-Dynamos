# AI-Powered Wellbeing Buffer

A local-first content buffer that sits between a social feed and the reader:
self-trained models blur harmful content *before* it renders, and a
pre-post check offers a rephrase before a risky draft goes out.

**Every model is trained from scratch by this repo's own scripts. Nothing
pretrained is loaded, nothing is fetched from a commercial LLM API, and
nothing is fetched from the internet at all at inference time except the
images the feed itself links to.**

## Repository layout

```
ai-wellbeing-layer/
├── backend/
│   ├── data/
│   │   ├── toxic_sample.csv         # 500-row labeled toxicity training set
│   │   └── topic_corpus.csv         # topic corpus for the trigger embedder
│   ├── image_features.py            # hand-engineered image feature extraction
│   ├── train_model.py               # trains custom_toxic_model.joblib
│   ├── train_nsfw_model.py          # trains nsfw_image_model.joblib
│   ├── train_trigger_embedder.py    # trains trigger_embedder.joblib
│   ├── main.py                      # FastAPI server + all endpoints
│   ├── requirements.txt
│   ├── custom_toxic_model.joblib    # pre-trained, ships ready to run
│   ├── nsfw_image_model.joblib      # pre-trained, ships ready to run
│   └── trigger_embedder.joblib      # pre-trained, ships ready to run
└── extension/
    ├── manifest.json
    ├── content.js
    ├── styles.css
    ├── popup.html
    └── popup.js
```

## The three self-trained models

| Feature | Model | Trained on | Architecture |
|---|---|---|---|
| Toxicity classification | `custom_toxic_model.joblib` | `data/toxic_sample.csv` (500 labeled comments) | TF-IDF + Logistic Regression |
| NSFW image tagging | `nsfw_image_model.joblib` | Procedurally generated synthetic images (`train_nsfw_model.py`) | Hand-engineered pixel/color/edge features (`image_features.py`) + Logistic Regression |
| Semantic trigger filter | `trigger_embedder.joblib` | `data/toxic_sample.csv` + `data/topic_corpus.csv` | TF-IDF + Truncated SVD (Latent Semantic Analysis) |

No Detoxify, no NudeNet, no sentence-transformers, no torchvision weights,
no OpenAI/Anthropic/Gemini calls anywhere in the stack.

**Why synthetic images for the NSFW classifier:** shipping or downloading a
real nudity dataset isn't appropriate for a public repo. Instead,
`train_nsfw_model.py` procedurally draws two classes of images whose
measurable properties mirror the actual signal being classified —
large/smooth/skin-tone-dominant regions (explicit) vs. varied, textured,
multi-color scenes that sometimes include a *small* skin-tone patch, like an
ordinary photo of a person (safe). This forces the model to learn "skin
coverage + smoothness," not just "any skin tone present," while keeping
data generation, feature extraction, training, and inference 100%
self-authored and reproducible with no external downloads. Swap
`build_dataset()` for a loader over a properly licensed dataset for a real
deployment — the model architecture and serving code don't need to change.

**Why LSA instead of pretrained sentence embeddings:** TF-IDF + Truncated
SVD is a classic, fully-from-scratch way to learn a topical vector space.
It's fit on our own corpus, so there are no pretrained weights anywhere in
the semantic-matching path — just our own text.

## 1. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

All three `.joblib` files are already included and pre-trained, so the
server starts instantly. If any of them is missing, `main.py` trains it
itself on startup by calling straight into that model's own training
script — the fallback for a missing model is "train it right now," never
"load a pretrained model instead."

To retrain any model manually:

```bash
python train_model.py              # toxicity classifier
python train_nsfw_model.py         # NSFW image classifier
python train_trigger_embedder.py   # semantic trigger embedder
```

Verify it's running: open `http://localhost:8000/` — you should see a
status block listing all three self-trained models.

## 2. Chrome extension setup

1. Go to `chrome://extensions`.
2. Enable **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select the `extension/` folder.
4. Open the extension popup to set your sensitivity threshold, trigger
   topics, and filter toggles.
5. Browse any site with the backend running — flagged posts blur
   automatically with a liftable "Unhide Content" warning, and risky
   drafts in comment boxes get an inline rephrase suggestion.

## Feature -> model mapping

- **Toxicity classification**: self-trained TF-IDF + LogisticRegression.
- **NSFW image tagging**: self-trained LogisticRegression over hand-engineered
  pixel features, trained on procedurally generated synthetic images.
- **NSFW text tagging**: reuses the self-trained toxicity score (at a
  stricter threshold) plus explicit-word pattern matching — no extra model.
- **Semantic trigger filter**: self-trained TF-IDF + LSA embedding space,
  cosine similarity against user-supplied trigger phrases.
- **Pre-post draft rephrasing**: deterministic rule-based de-escalation
  rewriter — no model, no API call.
- **Mood impact score**: closed-form formula over session stats — no ML
  needed.

## Demoing live self-training

Call `POST /train-custom-classifier` (no body) to re-run `train_model.py`
as a subprocess and hot-reload the resulting model into the running API —
useful for showing judges the classifier training live on stage.
