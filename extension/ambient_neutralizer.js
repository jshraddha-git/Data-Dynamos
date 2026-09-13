/**
 * ambient_neutralizer.js
 * ----------------------
 * User Experience USP: Ambient Neutralization (Calm Read).
 * On-device client-side neutralization engine.
 * Converts hostile social posts into calm, objective summaries right inside the browser.
 */

(() => {
  const INTENT_PATTERNS = [
    {
      regex: /\b(code|pr|repo|commit|design|architecture|implementation)\b.*?\b(garbage|trash|dogshit|shit|terrible|horrible|moron|idiot|clueless)\b/i,
      summary: "The reviewer expresses sharp technical disagreement with the implementation or design."
    },
    {
      regex: /\b(you|u)\s+(don't|dont|do not)\s+(know|understand)\s+(what|anything|how)\b.*?\b(moron|idiot|fool|stupid)\b/i,
      summary: "The author believes there is a misunderstanding of core technical concepts."
    },
    {
      regex: /\b(your\s+argument|what\s+you\s+said|your\s+point|this\s+take)\b.*?\b(idiotic|stupid|moronic|nonsense|bullshit|dumb|garbage)\b/i,
      summary: "The author strongly disputes the logic of the argument."
    },
    {
      regex: /\b(you\s+are|you're|ure)\s+(an?\s+)?(idiot|moron|dumbass|stupid|fool|clown|imbecile)\b/i,
      summary: "The poster expresses strong personal disagreement with the perspective presented."
    },
    {
      regex: /\b(shut\s+up|delete\s+your\s+account|stop\s+talking|get\s+lost|kill\s+yourself)\b/i,
      summary: "The speaker strongly rejects this viewpoint and urges disengagement from the topic."
    },
    {
      regex: /\b(scam|worthless|piece\s+of\s+shit|garbage|waste\s+of\s+money|trash)\b.*?\b(product|app|service|company|devs|team)\b/i,
      summary: "The user reports extreme dissatisfaction with the product, service, or team."
    }
  ];

  const TOXIC_TERMS = [
    /\b(idiot|idiotic|moron|moronic|stupid|dumb|dumbass|loser|pathetic|worthless|scum|filth|clown)\b/gi,
    /\b(kill\s+yourself|kys|die)\b/gi,
    /\b(shut\s+up|stfu|gtfo)\b/gi,
    /\b(delete\s+your\s+account)\b/gi,
    /\b(asshole|bitch|bastard|crap|garbage|trash|dogshit|shit|fuck|fucking|fucker)\b/gi,
    /\b(hate\s+you|despise\s+you)\b/gi
  ];

  function neutralize(text) {
    const clean = (text || "").trim();
    if (!clean) return { neutralized: "", calm_read: "" };

    for (const item of INTENT_PATTERNS) {
      if (item.regex.test(clean)) {
        return {
          neutralized: `[Calm Read: ${item.summary}]`,
          calm_read: item.summary,
          strategy: "pattern"
        };
      }
    }

    let sanitized = clean;
    for (const rx of TOXIC_TERMS) {
      sanitized = sanitized.replace(rx, "");
    }
    sanitized = sanitized.replace(/\s+/g, " ").replace(/\s+([.,!?])/g, "$1").trim();

    if (sanitized.length > 15 && sanitized.toLowerCase() !== clean.toLowerCase()) {
      return {
        neutralized: `[Calm Read: The author states: "${sanitized}"]`,
        calm_read: `The author states: "${sanitized}"`,
        strategy: "filtered"
      };
    }

    const fallback = "The author expresses emotionally charged disagreement regarding this topic.";
    return {
      neutralized: `[Calm Read: ${fallback}]`,
      calm_read: fallback,
      strategy: "fallback"
    };
  }

  const AmbientNeutralizerClient = { neutralize };

  if (typeof window !== "undefined") {
    window.AmbientNeutralizerClient = AmbientNeutralizerClient;
  }
  if (typeof self !== "undefined") {
    self.AmbientNeutralizerClient = AmbientNeutralizerClient;
  }
})();
