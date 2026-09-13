"""
ambient_neutralizer.py
----------------------
User Experience USP: Ambient Neutralization (Calm Read).
Rewrites hostile, toxic rants into calm, neutral, objective summaries without commercial LLM APIs.
Strips hostility and personal insults while preserving underlying debate, disagreement, or technical critique.
"""

import re
from typing import Dict, Any, List

# Hostile attack patterns and their calm, objective translations
INTENT_PATTERNS = [
    # 1. Technical / code competence attacks
    (
        r"\b(code|pr|repo|commit|design|architecture|implementation)\b.*?\b(garbage|trash|dogshit|shit|terrible|horrible|moron|idiot|clueless)\b",
        "The reviewer expresses sharp technical disagreement with the implementation or design."
    ),
    (
        r"\b(you|u)\s+(don't|dont|do not)\s+(know|understand)\s+(what|anything|how)\b.*?\b(moron|idiot|fool|stupid)\b",
        "The author believes there is a misunderstanding of core technical concepts."
    ),
    # 2. Logic / argument invalidation rants
    (
        r"\b(your\s+argument|what\s+you\s+said|your\s+point|this\s+take)\b.*?\b(idiotic|stupid|moronic|nonsense|bullshit|dumb|garbage)\b",
        "The author strongly disputes the logic of the argument."
    ),
    (
        r"\b(you\s+are|you're|ure)\s+(an?\s+)?(idiot|moron|dumbass|stupid|fool|clown|imbecile)\b",
        "The poster expresses strong personal disagreement with the perspective presented."
    ),
    # 3. Dismissal / silencing attacks
    (
        r"\b(shut\s+up|delete\s+your\s+account|stop\s+talking|get\s+lost|kill\s+yourself)\b",
        "The speaker strongly rejects this viewpoint and urges disengagement from the topic."
    ),
    # 4. Product / service dissatisfaction rants
    (
        r"\b(scam|worthless|piece\s+of\s+shit|garbage|waste\s+of\s+money|trash)\b.*?\b(product|app|service|company|devs|team)\b",
        "The user reports extreme dissatisfaction with the product, service, or team."
    ),
    # 5. Gatekeeping rants
    (
        r"\b(learn|read)\s+(the\s+basics|the\s+docs|how\s+to\s+code|documentation)\b",
        "The commenter suggests consulting reference documentation before continuing."
    ),
    # 6. Performance / gaming rants
    (
        r"\b(throw|threw|uninstall|feeding|noob|trash|terrible)\b.*?\b(game|match|ranked|player)\b",
        "The teammate expresses frustration with gameplay performance."
    )
]

# Profane / insult tokens to filter out in generative neutralizing
TOXIC_TERMS = [
    r"\b(idiot|idiotic|moron|moronic|stupid|dumb|dumbass|loser|pathetic|worthless|scum|filth|clown)\b",
    r"\b(kill\s+yourself|kys|die)\b",
    r"\b(shut\s+up|stfu|gtfo)\b",
    r"\b(delete\s+your\s+account)\b",
    r"\b(asshole|bitch|bastard|crap|garbage|trash|dogshit|shit|fuck|fucking|fucker)\b",
    r"\b(hate\s+you|despise\s+you)\b"
]


class AmbientNeutralizer:
    """Transforms aggressive feeds into calm, objective summaries."""

    def neutralize(self, text: str) -> Dict[str, Any]:
        text_clean = (text or "").strip()
        if not text_clean:
            return {
                "original": "",
                "neutralized": "",
                "strategy": "empty",
                "calm_read": ""
            }

        lowered = text_clean.lower()

        # 1. Pattern-based Intent Recognition
        for pattern, calm_summary in INTENT_PATTERNS:
            if re.search(pattern, lowered, re.IGNORECASE):
                return {
                    "original": text_clean,
                    "neutralized": f"[Calm Read: {calm_summary}]",
                    "strategy": "pattern_intent",
                    "calm_read": calm_summary
                }

        # 2. De-escalate via Toxic Term Stripping and Content Reconstruction
        sanitized = text_clean
        for t_regex in TOXIC_TERMS:
            sanitized = re.sub(t_regex, "", sanitized, flags=re.IGNORECASE)

        # Normalize whitespace and clean punctuation
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        sanitized = re.sub(r"\s+([.,!?])", r"\1", sanitized)
        sanitized = re.sub(r"([!?.])\1+", r"\1", sanitized)

        if len(sanitized) > 15 and sanitized.lower() != text_clean.lower():
            # If substantive informational text remains after removing insults
            neutral_version = f"[Calm Read: The author states: \"{sanitized}\"]"
            return {
                "original": text_clean,
                "neutralized": neutral_version,
                "strategy": "term_filtered",
                "calm_read": f"The author states: \"{sanitized}\""
            }

        # 3. Fallback General De-escalation
        summary = "The author expresses emotionally charged disagreement regarding this topic."
        return {
            "original": text_clean,
            "neutralized": f"[Calm Read: {summary}]",
            "strategy": "generic_deescalation",
            "calm_read": summary
        }


ambient_neutralizer = AmbientNeutralizer()
