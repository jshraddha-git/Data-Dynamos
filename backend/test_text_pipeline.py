import json
from main import score_toxicity, semantic_trigger_match, is_text_nsfw, suggest_rephrase, explain_toxicity

test_cases = [
    ("You are a pathetic idiot and nobody likes you", 0.5),
    ("Shut up and delete your account, you worthless loser", 0.5),
    ("Bro, you absolutely killed that guitar solo on stage tonight!", 0.5),
    ("Company announced massive layoffs and job cuts across the division", 0.5),
    ("What a wonderful day for a morning walk in the park", 0.5),
    ("u r a f*cking idiot go k!ll urself", 0.5),
    ("This movie ending has a major spoiler: the main hero dies in the climax", 0.5),
    ("I completely disagree with your macroeconomic analysis and methodology", 0.5)
]

triggers = ["layoffs", "spoilers", "crypto"]

print("============================================================")
print("VERIFYING TEXT PIPELINE END-TO-END")
print("============================================================")

for text, threshold in test_cases:
    tox_score, dv_meta = score_toxicity(text)
    is_tox = tox_score >= threshold
    is_nsfw = is_text_nsfw(text, tox_score)
    trig_hit, trig_name, trig_score = semantic_trigger_match(text, triggers)
    explanation = explain_toxicity(text) if is_tox else []

    action = "BLUR" if (is_tox or is_nsfw or trig_hit) else "SHOW"

    print(f"INPUT: \"{text}\"")
    print(f"  Action: [{action}] | Toxicity Score: {tox_score:.4f} | Is Toxic: {is_tox} | Text NSFW: {is_nsfw}")
    if trig_hit:
        print(f"  Trigger Match: '{trig_name}' (Semantic similarity: {trig_score:.3f})")
    if explanation:
        terms = ", ".join([f"{e['term']} (+{e['contribution']})" for e in explanation])
        print(f"  Explainability Attribution: {terms}")
    if is_tox:
        print(f"  Constructive Rephrase: \"{suggest_rephrase(text)}\"")
    print("-" * 60)
