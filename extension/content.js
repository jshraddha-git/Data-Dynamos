/**
 * content.js
 * ----------
 * Injected into every page. Responsibilities:
 *   1. Scan the DOM for "post-like" content blocks (using a MutationObserver
 *      so infinite-scroll feeds are handled), extract text + image URLs,
 *      and send them to the local backend's /analyze-post endpoint.
 *   2. Apply a CSS blur + liftable warning badge over anything flagged.
 *   3. Watch compose boxes (textareas / contenteditable) and run drafts
 *      through /check-draft, injecting an inline rephrase suggestion.
 *   4. Track session stats (blocked counts, session start time) in
 *      chrome.storage.local so the popup dashboard can read them.
 */

(() => {
  const API_BASE = "http://localhost:8000";

  const DEFAULT_SETTINGS = {
    userTriggers: [],
    toxicityThreshold: 0.5,
    nsfwFilterEnabled: true,
    draftCheckEnabled: true,
  };

  // In-memory cache of settings, refreshed from chrome.storage.local on load
  // and whenever the popup writes a change (via storage.onChanged).
  let settings = { ...DEFAULT_SETTINGS };

  // Track which DOM nodes we've already sent to the backend so we don't
  // re-analyze the same post on every mutation.
  const PROCESSED_ATTR = "data-wb-processed";
  const COMPOSE_ATTR = "data-wb-compose-attached";

  // Candidate selectors for "post-like" containers. Kept broad + generic so
  // the extension works across arbitrary sites, not just Twitter/Reddit.
  const POST_SELECTORS = [
    'article',
    '[data-testid="tweet"]',
    '[data-test-id="post-content"]',
    '[data-testid="post-container"]',
    '.Post',
    '.thing', // old reddit
  ].join(", ");

  const MIN_TEXT_LENGTH_TO_ANALYZE = 15;

  // ---------------------------------------------------------------------
  // Settings load + live sync with popup
  // ---------------------------------------------------------------------
  function loadSettings() {
    chrome.storage.local.get(DEFAULT_SETTINGS, (stored) => {
      settings = { ...DEFAULT_SETTINGS, ...stored };
    });
  }

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (key in settings) settings[key] = newValue;
    }
  });

  loadSettings();
  ensureSessionInitialized();

  // ---------------------------------------------------------------------
  // Session stat helpers (read/written by popup.js too)
  // ---------------------------------------------------------------------
  function ensureSessionInitialized() {
    chrome.storage.local.get(["sessionStart"], (data) => {
      if (!data.sessionStart) {
        chrome.storage.local.set({
          sessionStart: Date.now(),
          toxicBlockedCount: 0,
          nsfwBlockedCount: 0,
          triggerBlockedCount: 0,
        });
      }
    });
  }

  function incrementBlockedCounter(kind) {
    const key =
      kind === "toxic"
        ? "toxicBlockedCount"
        : kind === "nsfw"
        ? "nsfwBlockedCount"
        : "triggerBlockedCount";

    chrome.storage.local.get([key], (data) => {
      const current = data[key] || 0;
      chrome.storage.local.set({ [key]: current + 1 });
    });
  }

  // ---------------------------------------------------------------------
  // Post scanning + backend analysis
  // ---------------------------------------------------------------------
  function collectCandidateBlocks(root) {
    const blocks = new Set();

    if (root.matches && root.matches(POST_SELECTORS)) blocks.add(root);
    if (root.querySelectorAll) {
      root.querySelectorAll(POST_SELECTORS).forEach((el) => blocks.add(el));
    }

    // Fallback: reasonably-sized text-bearing <div>/<p> blocks that aren't
    // already inside something we've flagged, for sites with no semantic
    // post markup at all.
    if (root.querySelectorAll) {
      root.querySelectorAll("div, p").forEach((el) => {
        if (blocks.has(el)) return;
        const text = (el.innerText || "").trim();
        if (
          text.length >= MIN_TEXT_LENGTH_TO_ANALYZE &&
          el.children.length <= 6 &&
          !el.closest(`[${PROCESSED_ATTR}]`)
        ) {
          blocks.add(el);
        }
      });
    }

    return Array.from(blocks);
  }

  function extractImageUrls(block) {
    return Array.from(block.querySelectorAll("img"))
      .map((img) => img.src)
      .filter((src) => src && src.startsWith("http"))
      .slice(0, 5); // cap payload size
  }

  async function analyzeAndMaybeBlur(block) {
    if (block.hasAttribute(PROCESSED_ATTR)) return;
    block.setAttribute(PROCESSED_ATTR, "pending");

    const text = (block.innerText || "").trim();
    const imageUrls = settings.nsfwFilterEnabled ? extractImageUrls(block) : [];

    if (text.length < MIN_TEXT_LENGTH_TO_ANALYZE && imageUrls.length === 0) {
      block.setAttribute(PROCESSED_ATTR, "skipped");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/analyze-post`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          image_urls: imageUrls,
          user_triggers: settings.userTriggers,
          toxicity_threshold: settings.toxicityThreshold,
        }),
      });

      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const result = await res.json();

      block.setAttribute(PROCESSED_ATTR, "done");

      if (result.action === "blur") {
        applyBlurOverlay(block, result);
      }
    } catch (err) {
      // Backend unreachable (e.g. not running) — fail open, don't block content.
      console.warn("[WellbeingBuffer] analyze-post failed:", err);
      block.setAttribute(PROCESSED_ATTR, "error");
    }
  }

  // ---------------------------------------------------------------------
  // Blur overlay + liftable warning badge
  // ---------------------------------------------------------------------
  function classifyBadge(result) {
    if (result.is_toxic) {
      return { text: "Toxic Content Blocked", cssClass: "wb-badge-toxic" };
    }
    if (result.is_nsfw) {
      return { text: "NSFW Content Blocked", cssClass: "wb-badge-nsfw" };
    }
    if (result.trigger_matched) {
      return {
        text: `Topic Trigger Matched: '${result.matched_trigger}'`,
        cssClass: "wb-badge-trigger",
      };
    }
    return { text: "Content Blocked", cssClass: "wb-badge-toxic" };
  }

  function applyBlurOverlay(block, result) {
    const computedPosition = window.getComputedStyle(block).position;
    if (computedPosition === "static") {
      block.style.position = "relative";
    }

    block.classList.add("wb-blurred");

    const badgeInfo = classifyBadge(result);

    const overlay = document.createElement("div");
    overlay.className = "wb-overlay";

    const badge = document.createElement("div");
    badge.className = `wb-badge ${badgeInfo.cssClass}`;
    badge.textContent = badgeInfo.text;

    const reasonEl = document.createElement("div");
    reasonEl.className = "wb-reason";
    reasonEl.textContent = result.reason;

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "wb-unhide-btn";
    toggleBtn.textContent = "Unhide Content";
    toggleBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const isBlurred = block.classList.toggle("wb-blurred");
      toggleBtn.textContent = isBlurred ? "Unhide Content" : "Hide Content";
      overlay.classList.toggle("wb-overlay-hidden", !isBlurred);
    });

    overlay.appendChild(badge);
    overlay.appendChild(reasonEl);
    overlay.appendChild(toggleBtn);
    block.appendChild(overlay);

    // Record which counter this hit belongs to (priority: toxic > nsfw > trigger,
    // matching the same precedence used server-side to pick `reason`).
    if (result.is_toxic) incrementBlockedCounter("toxic");
    else if (result.is_nsfw) incrementBlockedCounter("nsfw");
    else if (result.trigger_matched) incrementBlockedCounter("trigger");
  }

  // ---------------------------------------------------------------------
  // Pre-post compose-box draft checking
  // ---------------------------------------------------------------------
  function findComposeElements(root) {
    const els = new Set();
    if (root.matches && root.matches('textarea, [contenteditable="true"]')) {
      els.add(root);
    }
    if (root.querySelectorAll) {
      root
        .querySelectorAll('textarea, [contenteditable="true"]')
        .forEach((el) => els.add(el));
    }
    return Array.from(els);
  }

  function getElementText(el) {
    return el.tagName === "TEXTAREA" ? el.value : el.innerText;
  }

  function attachComposeListener(el) {
    if (el.hasAttribute(COMPOSE_ATTR)) return;
    el.setAttribute(COMPOSE_ATTR, "true");

    let debounceTimer = null;

    el.addEventListener("input", () => {
      if (!settings.draftCheckEnabled) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => checkDraft(el), 600);
    });
  }

  async function checkDraft(el) {
    const draftText = (getElementText(el) || "").trim();
    removeDraftAlert(el);

    if (draftText.length < MIN_TEXT_LENGTH_TO_ANALYZE) return;

    try {
      const res = await fetch(`${API_BASE}/check-draft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ draft_text: draftText }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const result = await res.json();

      if (result.is_risky) {
        showDraftAlert(el, result);
      }
    } catch (err) {
      console.warn("[WellbeingBuffer] check-draft failed:", err);
    }
  }

  function removeDraftAlert(el) {
    const existing = el._wbAlertBox;
    if (existing && existing.parentNode) {
      existing.parentNode.removeChild(existing);
    }
    el._wbAlertBox = null;
  }

  function showDraftAlert(el, result) {
    const alertBox = document.createElement("div");
    alertBox.className = "wb-draft-alert";

    const title = document.createElement("div");
    title.className = "wb-draft-alert-title";
    title.textContent = `⚠ This post may come across as toxic (score ${result.toxicity_score.toFixed(
      2
    )})`;

    const suggestion = document.createElement("div");
    suggestion.className = "wb-draft-alert-suggestion";
    suggestion.textContent = result.suggestion || "";

    const useBtn = document.createElement("button");
    useBtn.className = "wb-draft-alert-use-btn";
    useBtn.textContent = "Use Suggestion";
    useBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (el.tagName === "TEXTAREA") {
        el.value = result.suggestion;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      } else {
        el.innerText = result.suggestion;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
      removeDraftAlert(el);
    });

    const dismissBtn = document.createElement("button");
    dismissBtn.className = "wb-draft-alert-dismiss-btn";
    dismissBtn.textContent = "Dismiss";
    dismissBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeDraftAlert(el);
    });

    alertBox.appendChild(title);
    alertBox.appendChild(suggestion);
    alertBox.appendChild(useBtn);
    alertBox.appendChild(dismissBtn);

    // Insert right after the compose element in the DOM.
    if (el.parentNode) {
      el.parentNode.insertBefore(alertBox, el.nextSibling);
    }
    el._wbAlertBox = alertBox;
  }

  // ---------------------------------------------------------------------
  // MutationObserver wiring
  // ---------------------------------------------------------------------
  function processNode(node) {
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    collectCandidateBlocks(node).forEach((block) => {
      // requestIdleCallback keeps scanning from janking the scroll feed;
      // falls back to setTimeout on browsers without it.
      const schedule = window.requestIdleCallback || ((cb) => setTimeout(cb, 50));
      schedule(() => analyzeAndMaybeBlur(block));
    });

    findComposeElements(node).forEach(attachComposeListener);
  }

  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach(processNode);
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // Initial pass over content already on the page at load time.
  processNode(document.body);
})();
