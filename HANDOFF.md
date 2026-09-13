# Handoff

## Current state
Repository initialized for **Can You Hack It (CYHI)** Track 3: **AI/ML - A Layer Between You and the Noise**.
Architecture comprises a FastAPI backend with self-trained models (Toxicity Classifier, Semantic Trigger Engine via LSA, and Multimodal Disparity Scorer) and a Chrome Manifest V3 extension (`extension/`) targeting multiple social platforms (Twitter, Reddit, Instagram, etc.).

## Works
- Self-trained dual-stream TF-IDF (word + char n-grams) + Logistic Regression toxicity classifier with linear feature attribution.
- LSA trigger matching (TF-IDF + TruncatedSVD 64d) for semantic user-defined topics without external LLM API dependency.
- Multi-platform DOM extraction adapters and content script injection logic.
- CYHI session logging configured (`cyhi-logs/` active for team Data Dynamos).

## Broken
- Backend local environment & dependencies need live run-time verification.
- End-to-end integration between the MV3 extension and local API endpoint requires live testing with browser feed feeds.

## Next 3 things
1. Verify Python backend environment & test model loading / inference endpoints.
2. Load and verify Chrome extension (MV3) background service worker and content scripts.
3. Validate real-time feed filtering, attribution chips, and user sensitivity calibration.

## Decisions (and why)
- **Zero commercial LLM API dependencies**: Strict compliance with CYHI Track 3 rule requiring majority self-trained models.
- **Dual-stream n-grams**: Catches leetspeak and obfuscated attacks without flagging benign colloquialisms.

## Don't retry
- Pure keyword matching (bypassed easily by leetspeak and produces high false positives on slang).
- Relying on heavy remote LLM inference for feed interception (violates track constraint and introduces latency).
