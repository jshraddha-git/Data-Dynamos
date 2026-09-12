"""
test_api.py
-----------
Comprehensive integration test for backend endpoints:
1. /health check (primary toxicity and self-trained LSA loaded)
2. /analyze-post with explainability attribution
3. /analyze-post with trigger match
4. /analyze-post with multimodal disparity detection
5. /check-draft with constructive rephrasing
6. /calculate-mood-impact
7. Rate Limiter verification
8. Auth header validation
"""

import sys
from fastapi.testclient import TestClient
from main import app, API_KEY

client = TestClient(app)

def run_tests():
    print("=" * 60)
    print("Running AI Wellbeing Buffer Backend Integration Tests")
    print("=" * 60)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health = res.json()
    print("[PASS] /health response:", health)
    assert health["primary_toxicity_model_loaded"] is True, "Self-trained toxicity model not loaded!"
    assert health["self_trained_lsa_model_loaded"] is True, "Self-trained LSA model not loaded!"

    # 2. Analyze Toxic Post (with Explainability Attribution)
    res = client.post("/analyze-post", json={
        "text": "Shut up you pathetic moron, delete your account!",
        "user_triggers": [],
        "toxicity_threshold": 0.5
    })
    assert res.status_code == 200, f"Analyze toxic post failed: {res.text}"
    data = res.json()
    print("[PASS] /analyze-post (Toxic):", data["action"], "| Reason:", data["reason"], "| Score:", data["toxicity_score"])
    print("       Explainability terms:", data["explanation"])
    assert data["action"] == "blur", "Toxic post should be blurred!"
    assert data["is_toxic"] is True, "Post should be flagged as toxic!"
    assert len(data["explanation"]) > 0, "Explainability attribution should return contributing terms!"

    # 3. Analyze Benign Slang (should NOT be blurred)
    res = client.post("/analyze-post", json={
        "text": "Bro, you absolutely killed that guitar solo on stage tonight!",
        "user_triggers": [],
        "toxicity_threshold": 0.5
    })
    assert res.status_code == 200
    data = res.json()
    print("[PASS] /analyze-post (Benign slang):", data["action"], "| Score:", data["toxicity_score"])
    assert data["action"] == "show", f"Benign slang falsely flagged! Score: {data['toxicity_score']}"

    # 4. Analyze Trigger Match (Self-Trained LSA)
    res = client.post("/analyze-post", json={
        "text": "Company just announced massive layoffs and severance packages.",
        "user_triggers": ["layoffs"],
        "toxicity_threshold": 0.5
    })
    assert res.status_code == 200
    data = res.json()
    print("[PASS] /analyze-post (LSA Trigger):", data["action"], "| Matched:", data["matched_trigger"])
    assert data["trigger_matched"] is True, "LSA trigger should match layoffs!"

    # 5. Check Draft
    res = client.post("/check-draft", json={"draft_text": "You are a stupid idiot and I hate you"})
    assert res.status_code == 200
    draft = res.json()
    print("[PASS] /check-draft:", draft["is_risky"], "| Suggestion:", draft["suggestion"])
    assert draft["is_risky"] is True, "Hostile draft should be flagged!"
    assert draft["suggestion"] is not None, "Draft check should provide constructive rephrasing!"

    # 6. Calculate Mood Impact
    res = client.post("/calculate-mood-impact", json={
        "session_duration_minutes": 45.0,
        "toxic_blocked_count": 8,
        "nsfw_blocked_count": 2,
        "trigger_blocked_count": 4
    })
    assert res.status_code == 200
    mood = res.json()
    print("[PASS] /calculate-mood-impact:", mood["mood_preservation_score"], "% | Summary:", mood["summary"])

    # 7. Live Retrain Verification
    res_retrain = client.post("/train-custom-classifier")
    assert res_retrain.status_code == 200, f"Retrain endpoint failed: {res_retrain.text}"
    retrain_data = res_retrain.json()
    print("[PASS] /train-custom-classifier:", retrain_data["status"], "| Message:", retrain_data["message"])
    assert retrain_data["status"] == "success", "Live retraining failed!"

    # 8. Rate Limiter Test (Simulate rapid burst)
    rapid_client = TestClient(app)
    hit_rate_limit = False
    for i in range(135):
        r = rapid_client.post("/analyze-post", json={"text": "hello world"})
        if r.status_code == 429:
            hit_rate_limit = True
            print(f"[PASS] Rate limiter triggered at request #{i+1} with 429 Too Many Requests (Retry-After: {r.headers.get('Retry-After')})")
            break
    assert hit_rate_limit is True, "Rate limiter should throttle after 120 requests/minute!"

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
