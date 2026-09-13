# Handoff

## Current state
AI Wellbeing Layer for **Can You Hack It (CYHI)** Track 3: **AI/ML - A Layer Between You and the Noise**.
Architecture comprises a FastAPI production backend running 100% self-trained models (Zero Commercial LLM APIs) and a Chrome Manifest V3 extension (`extension/`) supporting Reddit, Twitter/X, Instagram, TikTok, LinkedIn, YouTube, Facebook, and Generic web feeds.

## Works & USPs Implemented
1. **USP 1: Dual-Stream Multimodal Disparity Scorer (`backend/multimodal_scorer.py`)**:
   - Stream A (Vision): OpenCV + PyTorch extracts spatial gradient energy, color tension moments, and edge entropy.
   - Stream B (Text): 64d dense concept vectors from self-trained LSA manifold.
   - Resolves "malicious subtlety" / toxic memes via cross-modal cosine disparity.
2. **USP 2: Online SGD Personalization Engine (`backend/personalization_engine.py`)**:
   - Learns dynamically from implicit/explicit unhide ($y=0$) and rehide ($y=1$) user interactions.
   - Performs online SGD updates on personal bias and per-term vector shifts ($w_{\text{user}}$).
   - Real-time dashboard in `popup.html` / `popup.js` displaying learned bias and top tolerated vocabulary.
3. **USP 3: Dual-Vector Context Fusion (`backend/dual_vector_fusion.py`)**:
   - Decouples substantive topical discourse (politics, elections, layoffs, sensitive news) from personal emotional hostility.
   - Dampens false positive spikes on civil discussions when direct slur energy is near zero.
4. **Explainability Attribution Chips (USP 4)**:
   - Surfaces exact driving token attributions (e.g. `"idiot" (+0.243)`) inside the overlay.
5. **Sibling Wrapper Architecture (`.wb-post-wrapper`)**:
   - Fixed CSS blur leak: Blurred post container (`.wb-target-blurred`) and the overlay (`.wb-overlay`) are separate DOM siblings. Overlays, badges, and buttons remain 100% razor-sharp.

## Verified
- Automated test suite verifying `/health`, `/analyze-post`, `/personalize-feedback`, `/personalize-stats`, and `/personalize-reset`.
- Full synchronization between `C:\Users\Shraddha\Downloads\ai-wellbeing-layer-fixed` and `D:\CYHI\ai-wellbeing-layer-fixed`.

## Decisions (and why)
- **Zero commercial LLM API dependencies**: 100% compliance with CYHI Track 3 rule requiring majority self-trained models.
- **Offline Self-Trained LSA over HuggingFace**: Uses self-trained `custom_semantic_lsa.joblib` instead of remote HuggingFace Hub to avoid cold-start network hangs and satisfy offline hackathon requirements.
- **Online SGD Vector Shifts**: Stores per-user weight adjustments in local JSON profiles without leaking raw private browsing history.

## Don't retry
- Pure keyword blacklists (trivially bypassed by leetspeak and produces massive false alarms).
- Direct child insertion of overlay inside blurred nodes (CSS `filter: blur()` blurs all descendents).

