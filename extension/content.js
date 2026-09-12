/**
 * content.js
 * -----------
 * Runs on every page (see manifest.json host_permissions).
 *
 * Responsibilities:
 *   1. Scan the feed for post-like nodes as they're added (MutationObserver
 *      handles infinite-scroll feeds), extract text + image URLs, and send
 *      them to the backend BEFORE they're visually settled on screen.
 *   2. Apply a CSS blur + liftable overlay badge to anything flagged.
 *   3. Watch comment/post compose boxes and run a debounced pre-post
 *      toxicity check, injecting an inline rephrase suggestion.
 *   4. Track session stats (posts blocked, by category) for the mood
 *      dashboard in popup.js, persisted to chrome.storage.local.
 */

(() => {
  // Backend calls now go through background.js (see analyzePost/checkDraft
  // below) — content scripts injected into https:// pages get blocked by
  // Chrome's Private Network Access policy when fetching localhost directly.

  // ---------------------------------------------------------------------
  // Config loaded from chrome.storage.local (set via popup.js)
  // ---------------------------------------------------------------------
  const DEFAULT_SETTINGS = {
    toxicityThreshold: 0.5,
    userTriggers: [],
    nsfwFilteringEnabled: true,
    draftCheckEnabled: true,
    extensionEnabled: true,
  };

  let settings = { ...DEFAULT_SETTINGS };

  function loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.local.get(
        ["toxicityThreshold", "userTriggers", "nsfwFilteringEnabled", "draftCheckEnabled", "extensionEnabled"],
        (stored) => {
          settings = {
            toxicityThreshold: stored.toxicityThreshold ?? DEFAULT_SETTINGS.toxicityThreshold,
            userTriggers: stored.userTriggers ?? DEFAULT_SETTINGS.userTriggers,
            nsfwFilteringEnabled: stored.nsfwFilteringEnabled ?? DEFAULT_SETTINGS.nsfwFilteringEnabled,
            draftCheckEnabled: stored.draftCheckEnabled ?? DEFAULT_SETTINGS.draftCheckEnabled,
            extensionEnabled: stored.extensionEnabled ?? DEFAULT_SETTINGS.extensionEnabled,
          };
          resolve(settings);
        }
      );
    });
  }

  // Re-sync settings whenever the popup changes them
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (key in settings) settings[key] = newValue;
    }
  });

  // ---------------------------------------------------------------------
  // Session stats (for the Mood & Wellbeing dashboard)
  // ---------------------------------------------------------------------
  const sessionStart = Date.now();
  let sessionStats = { toxicBlocked: 0, nsfwBlocked: 0, triggerBlocked: 0 };

  function persistSessionStats() {
    chrome.storage.local.set({
      sessionStats,
      sessionStartTimestamp: sessionStart,
    });
  }

  // ---------------------------------------------------------------------
  // Backend calls
  // ---------------------------------------------------------------------
  async function analyzePost(text, imageUrls) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "wb-analyze-post",
        payload: {
          text: text || "",
          image_urls: settings.nsfwFilteringEnabled ? imageUrls : [],
          user_triggers: settings.userTriggers,
          toxicity_threshold: settings.toxicityThreshold,
        },
      });
      if (!response || !response.ok) throw new Error(response ? response.error : "no response");
      return response.data;
    } catch (err) {
      console.warn("[Wellbeing Buffer] analyze-post failed (is the backend running on :8000?)", err);
      return null;
    }
  }

  async function checkDraft(draftText) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "wb-check-draft",
        payload: { draft_text: draftText },
      });
      if (!response || !response.ok) throw new Error(response ? response.error : "no response");
      return response.data;
    } catch (err) {
      console.warn("[Wellbeing Buffer] check-draft failed", err);
      return null;
    }
  }

  // ---------------------------------------------------------------------
  // DOM: identifying "post-like" nodes generically across platforms
  // ---------------------------------------------------------------------
  // Heuristic selector list — broad enough to work across Twitter/X, Reddit,
  // and generic article/comment containers without a platform-specific
  // scraper for every site.
  const POST_SELECTORS = [
    "article",
    "[data-testid='tweet']",
    "[data-testid='cellInnerDiv']",
    "shreddit-comment",
    "shreddit-post",
    ".Comment",
    "[role='article']",
  ];

  const processedNodes = new WeakSet();

  function extractTextAndImages(node) {
    const text = (node.innerText || "").trim().slice(0, 2000);
    const images = Array.from(node.querySelectorAll("img"))
      .map((img) => img.src)
      .filter((src) => src && src.startsWith("http"))
      .slice(0, 5); // cap per-post to keep requests light
    return { text, images };
  }

  function buildOverlay(reason, badgeClass) {
    const overlay = document.createElement("div");
    overlay.className = "wb-overlay";

    const badge = document.createElement("span");
    badge.className = `wb-badge ${badgeClass}`;
    badge.textContent = reason;

    const button = document.createElement("button");
    button.className = "wb-unhide-btn";
    button.textContent = "Unhide Content";

    overlay.appendChild(badge);
    overlay.appendChild(button);
    return { overlay, button };
  }

  function badgeClassFor(result) {
    if (result.is_toxic) return "wb-badge-toxic";
    if (result.is_nsfw) return "wb-badge-nsfw";
    if (result.trigger_matched) return "wb-badge-trigger";
    return "wb-badge-toxic";
  }

  function applyBlur(node, result) {
    // Wrap so the overlay can be absolutely positioned relative to the post
    if (getComputedStyle(node).position === "static") {
      node.style.position = "relative";
    }
    node.classList.add("wb-wrapper", "wb-blurred");

    const { overlay, button } = buildOverlay(result.reason, badgeClassFor(result));
    node.appendChild(overlay);

    let revealed = false;
    button.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      revealed = !revealed;
      node.classList.toggle("wb-revealed", revealed);
      overlay.classList.toggle("wb-hidden", revealed);
      button.textContent = revealed ? "Hide Again" : "Unhide Content";
    });

    // Update session stats once per node
    if (result.is_toxic) sessionStats.toxicBlocked += 1;
    if (result.is_nsfw) sessionStats.nsfwBlocked += 1;
    if (result.trigger_matched) sessionStats.triggerBlocked += 1;
    persistSessionStats();
  }

  async function processNode(node) {
    if (!settings.extensionEnabled) return;
    if (processedNodes.has(node)) return;
    processedNodes.add(node);

    const { text, images } = extractTextAndImages(node);
    if (!text && images.length === 0) return;

    const result = await analyzePost(text, images);
    if (result && result.action === "blur") {
      applyBlur(node, result);
    }
  }

  function scanForPosts(root = document) {
    const nodes = new Set();
    for (const selector of POST_SELECTORS) {
      root.querySelectorAll(selector).forEach((n) => nodes.add(n));
    }
    nodes.forEach((node) => processNode(node));
  }

  // ---------------------------------------------------------------------
  // MutationObserver — handles infinite-scroll feeds
  // ---------------------------------------------------------------------
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach((added) => {
        if (added.nodeType !== Node.ELEMENT_NODE) return;
        scanForPosts(added.parentNode ? added : document);
        // Also check the added node itself against selectors directly
        for (const selector of POST_SELECTORS) {
          if (added.matches && added.matches(selector)) {
            processNode(added);
          }
        }
      });
    }
  });

  function startObserving() {
    observer.observe(document.body, { childList: true, subtree: true });
    scanForPosts(); // initial pass for content already on the page
  }

  // ---------------------------------------------------------------------
  // Pre-post compose check (debounced)
  // ---------------------------------------------------------------------
  function debounce(fn, delay) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function findOrCreateWarningBox(inputEl) {
    let box = inputEl._wbWarningBox;
    if (box && document.body.contains(box)) return box;

    box = document.createElement("div");
    box.className = "wb-draft-warning";
    box.style.display = "none";
    inputEl._wbWarningBox = box;

    // Insert right after the input element (or its closest block parent)
    const parent = inputEl.parentElement || document.body;
    parent.insertBefore(box, inputEl.nextSibling);
    return box;
  }

  function renderDraftWarning(inputEl, result) {
    const box = findOrCreateWarningBox(inputEl);
    if (!result || !result.is_risky) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
    box.innerHTML = `<strong>⚠ This might come across as harsh</strong> (toxicity score: ${(
      result.toxicity_score * 100
    ).toFixed(0)}%)<span class="wb-suggestion">Suggested rephrase: "${result.suggestion}"</span>`;
  }

  const debouncedDraftCheck = debounce(async (inputEl) => {
    if (!settings.draftCheckEnabled) return;
    const text = inputEl.value !== undefined ? inputEl.value : inputEl.innerText;
    if (!text || text.trim().length < 4) {
      renderDraftWarning(inputEl, null);
      return;
    }
    const result = await checkDraft(text);
    renderDraftWarning(inputEl, result);
  }, 600);

  function attachComposeListeners(root = document) {
    const composeSelectors = [
      "textarea",
      "[contenteditable='true']",
      "input[type='text']",
      "div[role='textbox']",
    ];
    composeSelectors.forEach((selector) => {
      root.querySelectorAll(selector).forEach((el) => {
        if (el._wbListenerAttached) return;
        el._wbListenerAttached = true;
        el.addEventListener("input", () => debouncedDraftCheck(el));
      });
    });
  }

  const composeObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.addedNodes.length > 0) {
        // Re-scan the whole document for new compose boxes; cheap because
        // attachComposeListeners skips elements it has already wired up.
        attachComposeListeners(document);
        break;
      }
    }
  });

  function startComposeWatching() {
    attachComposeListeners();
    composeObserver.observe(document.body, { childList: true, subtree: true });
  }

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  (async function init() {
    await loadSettings();
    if (!document.body) {
      window.addEventListener("DOMContentLoaded", () => {
        startObserving();
        startComposeWatching();
      });
    } else {
      startObserving();
      startComposeWatching();
    }
  })();
})();
