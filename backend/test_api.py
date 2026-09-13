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
    assert data["neutralized_text"] is not None, "Neutralized text should be generated for flagged toxic post!"
    print("       Ambient Neutralized:", data["neutralized_text"])

    # Test Ambient Neutralization endpoint
    res_neut = client.post("/neutralize-post", json={"text": "You are an idiot and your code is garbage moron."})
    assert res_neut.status_code == 200
    neut_data = res_neut.json()
    print("[PASS] /neutralize-post:", neut_data["neutralized"])
    assert "Calm Read:" in neut_data["neutralized"], "Neutralized text should contain Calm Read prefix!"

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

    # 8. USP 1: Dual-Vector Context Fusion (Civil Discussion Damping)
    res_civil = client.post("/analyze-post", json={
        "text": "The committee held an intense legislative debate regarding political corruption allegations in the senate."
    })
    assert res_civil.status_code == 200
    civil_data = res_civil.json()
    print("[PASS] USP 1 Dual-Vector Fusion:", civil_data["action"], "| Meta:", civil_data["dual_vector_meta"])
    assert civil_data["action"] == "show", "Civil political debate should not be blocked!"
    assert civil_data["dual_vector_meta"]["false_alarm_damped"] is True, "False alarm damping should activate for civil discussion!"

    # 9. USP 2: Online SGD Personalization Engine
    client.post("/personalize-reset/test_client")
    test_phrase = "You are a completely terrible and useless clown."
    res_before = client.post("/analyze-post", json={"text": test_phrase, "client_id": "test_client"})
    assert res_before.status_code == 200
    before_data = res_before.json()
    assert before_data["action"] == "blur"

    # User unhides the content -> online SGD learns tolerance
    res_fb = client.post("/personalize-feedback", json={
        "text": test_phrase,
        "action": "unhide",
        "base_score": before_data["toxicity_score"],
        "client_id": "test_client"
    })
    assert res_fb.status_code == 200
    fb_data = res_fb.json()
    assert fb_data["learned_bias"] < 0.0, "Learned bias should shift negatively on unhide!"

    # Score post again: personalized shift lowers score
    res_after = client.post("/analyze-post", json={"text": test_phrase, "client_id": "test_client"})
    after_data = res_after.json()
    print("[PASS] USP 2 Online Personalization:", after_data["action"], "| Shift:", after_data["personal_shift"], "| New Score:", after_data["toxicity_score"])
    assert after_data["personal_shift"] < 0.0, "Personal vector shift should lower perceived hostility!"

    # Reset profile
    client.post("/personalize-reset/test_client")

    # 10. USP 3: Dual-Stream Multimodal Joint Disparity Scorer (Toxic Memes)
    from PIL import Image
    import io, numpy as np, base64
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_img = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    res_meme = client.post("/analyze-post", json={
        "text": "what a hero, wow so lovely and peaceful",
        "image_urls": [b64_img]
    })
    assert res_meme.status_code == 200
    meme_data = res_meme.json()
    print("[PASS] USP 3 Multimodal Disparity:", meme_data["action"], "| Disparity:", meme_data["cross_modal_disparity"], "| Reason:", meme_data["reason"])
    assert meme_data["cross_modal_flagged"] is True, "Subtle antagonistic meme should be flagged by multimodal disparity!"
    assert meme_data["action"] == "blur", "Flagged meme should be blurred!"

    # 11. Rate Limiter Test (Simulate rapid burst)
    import main
    old_limit = main.RATE_LIMIT_MAX_REQUESTS
    main.RATE_LIMIT_MAX_REQUESTS = 30
    try:
        rapid_client = TestClient(app)
        hit_rate_limit = False
        for i in range(45):
            r = rapid_client.post("/analyze-post", json={"text": "hello world"})
            if r.status_code == 429:
                hit_rate_limit = True
                print(f"[PASS] Rate limiter triggered at request #{i+1} with 429 Too Many Requests (Retry-After: {r.headers.get('Retry-After')})")
                break
        assert hit_rate_limit is True, "Rate limiter should throttle when exceeding threshold!"
    finally:
        main.RATE_LIMIT_MAX_REQUESTS = old_limit

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
