/**
 * content.js
 * -----------
 * Production content script for AI Wellbeing Buffer.
 *
 * Core Capabilities:
 *   1. Dynamically detects the host platform via WellbeingAdapterRegistry and
 *      applies platform-tailored selectors for Twitter/X, Reddit, Instagram,
 *      Facebook, TikTok, LinkedIn, YouTube, or Generic fallback.
 *   2. Extracts text, image URLs, and video canvas frame snapshots for multimodal
 *      and NSFW inspection.
 *   3. Implements Adaptive Personal Sensitivity: learns from the user's implicit
 *      unhide / rehide feedback to dynamically calibrate filtering sensitivity.
 *   4. Displays Explainability Attribution: surfaces top driving keywords from the
 *      self-trained model directly in the liftable post overlay.
 *   5. Monitors compose boxes with debounced draft checks and constructive rephrasing.
 */

(() => {
  // ---------------------------------------------------------------------
  // Config & State
  // ---------------------------------------------------------------------
  const DEFAULT_SETTINGS = {
    toxicityThreshold: 0.5,
    userTriggers: ["spoilers", "layoffs"],
    nsfwFilteringEnabled: true,
    draftCheckEnabled: true,
    extensionEnabled: true,
    adaptiveSensitivityEnabled: true,
    learnedSensitivityAdjustment: 0.0,
  };

  let settings = { ...DEFAULT_SETTINGS };
  let activeAdapter = null;

  function loadSettings() {
    return new Promise((resolve) => {
      chrome.storage.local.get(Object.keys(DEFAULT_SETTINGS), (stored) => {
        settings = { ...DEFAULT_SETTINGS, ...stored };
        resolve(settings);
      });
    });
  }

  // Re-sync settings whenever popup updates them
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    for (const [key, { newValue }] of Object.entries(changes)) {
      if (key in settings) settings[key] = newValue;
    }
  });

  // ---------------------------------------------------------------------
  // Session Stats
  // ---------------------------------------------------------------------
  const sessionStart = Date.now();
  let sessionStats = { toxicBlocked: 0, nsfwBlocked: 0, triggerBlocked: 0, memeBlocked: 0 };

  function persistSessionStats() {
    chrome.storage.local.set({
      sessionStats,
      sessionStartTimestamp: sessionStart,
    });
  }

  // ---------------------------------------------------------------------
  // Adaptive Personal Sensitivity Engine (USP 2)
  // ---------------------------------------------------------------------
  function getEffectiveThreshold() {
    if (!settings.adaptiveSensitivityEnabled) {
      return settings.toxicityThreshold;
    }
    const adjusted = settings.toxicityThreshold + (settings.learnedSensitivityAdjustment || 0.0);
    // Clamp to valid range [0.15, 0.95]
    return Math.max(0.15, Math.min(0.95, adjusted));
  }

  function recordUserFeedback(score, action, postText = "") {
    if (!settings.adaptiveSensitivityEnabled) return;

    // Dispatch feedback to backend online SGD personalization engine
    if (postText) {
      chrome.runtime.sendMessage({
        type: "wb-personalize-feedback",
        payload: {
          text: postText,
          action: action,
          base_score: score || 0.5,
          client_id: "default",
        },
      }).catch((err) => console.warn("[Wellbeing Buffer] Personalize feedback failed:", err));
    }

    chrome.storage.local.get(["learnedSensitivityAdjustment", "unhideCount"], (data) => {
      let adj = data.learnedSensitivityAdjustment || 0.0;
      let count = (data.unhideCount || 0) + 1;

      // If user unhides borderline content (0.40 - 0.80), increase tolerance
      if (action === "unhide") {
        if (score >= 0.40 && score <= 0.80) {
          adj = Math.min(0.20, adj + 0.02);
        }
      } else if (action === "rehide") {
        // User confirmed post was indeed unwanted, decrease tolerance
        adj = Math.max(-0.20, adj - 0.03);
      }

      chrome.storage.local.set({
        learnedSensitivityAdjustment: parseFloat(adj.toFixed(3)),
        unhideCount: count,
      });
    });
  }

  // ---------------------------------------------------------------------
  // Backend Communication
  // ---------------------------------------------------------------------
  async function analyzePost(text, mediaUrls) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "wb-analyze-post",
        payload: {
          text: text || "",
          image_urls: settings.nsfwFilteringEnabled ? mediaUrls : [],
          user_triggers: settings.userTriggers,
          toxicity_threshold: getEffectiveThreshold(),
        },
      });
      if (!response || !response.ok) throw new Error(response ? response.error : "no response");
      return response.data;
    } catch (err) {
      console.warn("[Wellbeing Buffer] analyze-post failed:", err);
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
      console.warn("[Wellbeing Buffer] check-draft failed:", err);
      return null;
    }
  }

  // ---------------------------------------------------------------------
  // DOM Overlay & Explainability Rendering (USP 1)
  // ---------------------------------------------------------------------
  function badgeClassFor(result) {
    if (result.cross_modal_flagged) return "wb-badge-meme";
    if (result.is_toxic) return "wb-badge-toxic";
    if (result.is_nsfw) return "wb-badge-nsfw";
    if (result.trigger_matched) return "wb-badge-trigger";
    return "wb-badge-toxic";
  }

  function buildOverlay(result) {
    const overlay = document.createElement("div");
    overlay.className = "wb-overlay";

    const badge = document.createElement("span");
    badge.className = `wb-badge ${badgeClassFor(result)}`;
    badge.textContent = result.reason;
    overlay.appendChild(badge);

    // Explainability Attribution Chip (USP 1)
    if (result.explanation && result.explanation.length > 0) {
      const expContainer = document.createElement("div");
      expContainer.className = "wb-explain-box";

      const expToggle = document.createElement("button");
      expToggle.type = "button";
      expToggle.className = "wb-explain-toggle";
      expToggle.textContent = "ℹ Why was this flagged?";

      const expContent = document.createElement("div");
      expContent.className = "wb-explain-content";
      expContent.style.display = "none";

      const termsList = result.explanation
        .map((e) => `<span class="wb-term-tag">"${e.term}" (+${e.contribution})</span>`)
        .join(" ");
      expContent.innerHTML = `<div class="wb-explain-desc">Model attribution signals:</div>${termsList}`;

      expToggle.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const isShown = expContent.style.display === "block";
        expContent.style.display = isShown ? "none" : "block";
        expToggle.textContent = isShown ? "ℹ Why was this flagged?" : "▲ Hide explanation";
      });

      expContainer.appendChild(expToggle);
      expContainer.appendChild(expContent);
      overlay.appendChild(expContainer);
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "wb-unhide-btn";
    button.textContent = "Unhide Content";
    overlay.appendChild(button);

    return { overlay, button };
  }

  function applyBlur(node, result, postText = "") {
    if (!node || node._wbBlurred || !node.parentNode) return;
    node._wbBlurred = true;

    // Wrap node so the overlay is an external sibling, never affected by post blur
    let wrapper = node.parentElement;
    if (!wrapper || !wrapper.classList.contains("wb-post-wrapper")) {
      wrapper = document.createElement("div");
      wrapper.className = "wb-post-wrapper";
      wrapper.style.cssText = "position: relative !important; display: block !important; width: 100% !important; margin: 0 !important; padding: 0 !important;";
      node.parentNode.insertBefore(wrapper, node);
      wrapper.appendChild(node);
    }

    node.classList.add("wb-target-blurred");

    const { overlay, button } = buildOverlay(result);
    wrapper.appendChild(overlay);

    let revealed = false;
    button.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      revealed = !revealed;
      node.classList.toggle("wb-target-blurred", !revealed);
      overlay.classList.toggle("wb-hidden", revealed);
      button.textContent = revealed ? "Hide Again" : "Unhide Content";

      // Trigger personal adaptation update
      recordUserFeedback(result.toxicity_score, revealed ? "unhide" : "rehide", postText);
    });

    // Update session metrics
    if (result.cross_modal_flagged) sessionStats.memeBlocked += 1;
    if (result.is_toxic) sessionStats.toxicBlocked += 1;
    if (result.is_nsfw) sessionStats.nsfwBlocked += 1;
    if (result.trigger_matched) sessionStats.triggerBlocked += 1;
    persistSessionStats();
  }

  // ---------------------------------------------------------------------
  // Feed Scanning via Platform Adapters
  // ---------------------------------------------------------------------
  const processedNodes = new WeakSet();

  async function processNode(node, retryCount = 0) {
    if (!settings.extensionEnabled || !node) return;
    if (processedNodes.has(node)) return;

    if (!activeAdapter) {
      activeAdapter = window.WellbeingAdapterRegistry.getActiveAdapter();
    }

    const postData = activeAdapter.extractPostData(node);

    // If node was just mounted and content hasn't rendered yet, retry after short delay
    if (!postData.text && postData.media.length === 0) {
      if (retryCount < 4) {
        setTimeout(() => {
          if (!processedNodes.has(node)) {
            processNode(node, retryCount + 1);
          }
        }, 200 * (retryCount + 1));
      }
      return;
    }

    processedNodes.add(node);

    const result = await analyzePost(postData.text, postData.media);
    if (!result) {
      // Backend temporarily unreachable or rate limited; allow future scans to retry
      processedNodes.delete(node);
      return;
    }

    if (result.action === "blur") {
      applyBlur(node, result, postData.text);
    }
  }

  function scanForPosts(root = document) {
    if (!activeAdapter) {
      activeAdapter = window.WellbeingAdapterRegistry.getActiveAdapter();
    }
    const selector = activeAdapter.getPostSelector();
    try {
      // If root itself matches the selector, process it directly
      if (root !== document && root.matches && root.matches(selector)) {
        processNode(root);
      }
      const nodes = root.querySelectorAll(selector);
      nodes.forEach((node) => processNode(node));
    } catch (err) {
      console.warn("[Wellbeing Buffer] Selector query error:", err);
    }
  }

  // ---------------------------------------------------------------------
  // MutationObserver for Infinite-Scroll Feeds
  // ---------------------------------------------------------------------
  let scanDebounceTimer = null;
  const feedObserver = new MutationObserver((mutations) => {
    if (!activeAdapter) {
      activeAdapter = window.WellbeingAdapterRegistry.getActiveAdapter();
    }
    const selector = activeAdapter.getPostSelector();

    let foundNew = false;
    for (const mutation of mutations) {
      for (const added of mutation.addedNodes) {
        if (added.nodeType !== Node.ELEMENT_NODE) continue;
        foundNew = true;
        if (added.matches && added.matches(selector)) {
          processNode(added);
        }
      }
    }

    if (foundNew) {
      if (scanDebounceTimer) clearTimeout(scanDebounceTimer);
      scanDebounceTimer = setTimeout(() => scanForPosts(), 250);
    }
  });

  function startFeedObserving() {
    feedObserver.observe(document.body, { childList: true, subtree: true });
    scanForPosts();
  }

  // ---------------------------------------------------------------------
  // Pre-Post Compose Watcher
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
    const text = activeAdapter.getDraftText(inputEl);
    if (!text || text.trim().length < 4) {
      renderDraftWarning(inputEl, null);
      return;
    }
    const result = await checkDraft(text);
    renderDraftWarning(inputEl, result);
  }, 500);

  function attachComposeListeners(root = document) {
    if (!activeAdapter) {
      activeAdapter = window.WellbeingAdapterRegistry.getActiveAdapter();
    }
    const composeSelector = activeAdapter.getComposeSelector();
    root.querySelectorAll(composeSelector).forEach((el) => {
      if (el._wbListenerAttached) return;
      el._wbListenerAttached = true;
      el.addEventListener("input", () => debouncedDraftCheck(el));
    });
  }

  const composeObserver = new MutationObserver(() => {
    attachComposeListeners(document);
  });

  function startComposeWatching() {
    attachComposeListeners();
    composeObserver.observe(document.body, { childList: true, subtree: true });
  }

  // ---------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------
  (async function init() {
    await loadSettings();
    activeAdapter = window.WellbeingAdapterRegistry.getActiveAdapter();
    console.log(`[AI Wellbeing Buffer] Initialized on ${window.location.hostname} using ${activeAdapter.name} adapter`);

    if (!document.body) {
      window.addEventListener("DOMContentLoaded", () => {
        startFeedObserving();
        startComposeWatching();
      });
    } else {
      startFeedObserving();
      startComposeWatching();
    }
  })();
})();
