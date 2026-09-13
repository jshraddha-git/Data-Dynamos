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
    autoNeutralizeEnabled: false,
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
  // Adaptive Personal Sensitivity & IndexedDB On-Device Shifts (USP 1)
  // ---------------------------------------------------------------------
  function getEffectiveThreshold() {
    return settings.toxicityThreshold || 0.5;
  }

  function recordUserFeedback(score, action, postText = "") {
    if (!settings.adaptiveSensitivityEnabled) return;

    if (postText) {
      // 1. On-Device Model Adaptation via IndexedDB
      chrome.runtime.sendMessage({
        type: "wb-db-record-shift",
        payload: {
          text: postText,
          action: action,
          baseScore: score || 0.5,
        },
      }).catch((err) => console.warn("[Wellbeing Buffer] IndexedDB record error:", err));

      // 2. Dispatch feedback to backend online SGD engine
      chrome.runtime.sendMessage({
        type: "wb-personalize-feedback",
        payload: {
          text: postText,
          action: action,
          base_score: score || 0.5,
          client_id: "default",
        },
      })
        .then((res) => {
          if (res && res.ok && res.data) {
            const learnedBias = res.data.learned_bias || 0.0;
            chrome.storage.local.set({
              learnedSensitivityAdjustment: learnedBias,
            });
          }
        })
        .catch((err) => console.warn("[Wellbeing Buffer] Personalize feedback failed:", err));
    }

    chrome.storage.local.get(["unhideCount"], (data) => {
      const count = (data.unhideCount || 0) + (action === "unhide" ? 1 : 0);
      chrome.storage.local.set({ unhideCount: count });
    });
  }

  // ---------------------------------------------------------------------
  // Backend Communication with On-Device Vector Shifts
  // ---------------------------------------------------------------------
  async function analyzePost(text, mediaUrls) {
    try {
      // 1. Query on-device IndexedDB vector shift
      let localShift = 0.0;
      let localTerms = [];
      try {
        const dbRes = await chrome.runtime.sendMessage({
          type: "wb-db-get-shift",
          payload: { text: text || "" },
        });
        if (dbRes && dbRes.ok && dbRes.data) {
          localShift = dbRes.data.shift || 0.0;
          localTerms = dbRes.data.matchedTerms || [];
        }
      } catch (_) {}

      // 2. Query backend ML inference
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

      const data = response.data;
      data.on_device_shift = localShift;
      data.on_device_terms = localTerms;

      // 3. Apply on-device IndexedDB vector shift if present (guardrailed for safety)
      if (localShift !== 0.0 && data.toxicity_score < 0.65) {
        data.toxicity_score = Math.max(0.0, Math.min(1.0, data.toxicity_score + localShift));
        data.is_toxic = data.toxicity_score >= getEffectiveThreshold();
        data.action = (data.is_toxic || data.is_nsfw || data.trigger_matched || data.cross_modal_flagged) ? "blur" : "show";
      }

      return data;
    } catch (err) {
      // Backend temporarily unreachable or rate limited; allow future scans to retry cleanly
      return null;
    }
  }

  async function checkDraft(draftText) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "wb-check-draft",
        payload: { draft_text: draftText },
      });
      if (!response || !response.ok) return null;
      return response.data;
    } catch (err) {
      // Fail silently without dumping warning to Chrome Extensions error tab
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

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function buildOverlay(result, postText = "") {
    const overlay = document.createElement("div");
    overlay.className = "wb-overlay";

    const badge = document.createElement("span");
    badge.className = `wb-badge ${badgeClassFor(result)}`;
    badge.textContent = result.reason;
    overlay.appendChild(badge);

    // -------------------------------------------------------------------
    // User Experience USP: Ambient Neutralization (Calm Read)
    // -------------------------------------------------------------------
    let neutralizedText = result.neutralized_text;
    if (!neutralizedText && window.AmbientNeutralizerClient) {
      neutralizedText = window.AmbientNeutralizerClient.neutralize(postText).neutralized;
    }
    if (!neutralizedText) {
      neutralizedText = "[Calm Read: The author expresses strong personal disagreement with this viewpoint.]";
    }

    const neutContainer = document.createElement("div");
    neutContainer.className = "wb-neutralized-container";
    neutContainer.style.display = "none";
    neutContainer.innerHTML = `
      <div class="wb-neutralized-header">🌿 Ambient Neutralized Summary (Calm Read)</div>
      <div class="wb-neutralized-body">${escapeHtml(neutralizedText)}</div>
    `;
    overlay.appendChild(neutContainer);

    // -------------------------------------------------------------------
    // Context Analysis & Explainability Radar (USP 1 & USP 3)
    // -------------------------------------------------------------------
    const expContainer = document.createElement("div");
    expContainer.className = "wb-explain-box";

    const expToggle = document.createElement("button");
    expToggle.type = "button";
    expToggle.className = "wb-explain-toggle";
    expToggle.textContent = "ℹ Why was this flagged? (Model Radar)";

    const expContent = document.createElement("div");
    expContent.className = "wb-explain-content";
    expContent.style.display = "none";

    let detailsHtml = "";

    // 1. Dual-Vector Subspace Projection
    if (result.dual_vector_meta && result.dual_vector_meta.topic_intensity !== undefined) {
      const meta = result.dual_vector_meta;
      const topicPct = Math.round((meta.topic_intensity || 0) * 100);
      const affectPct = Math.round((meta.affect_ratio || 0) * 100);
      const slurEnergy = meta.direct_hostility_energy || 0.0;
      detailsHtml += `
        <div class="wb-radar-card">
          <div class="wb-radar-title">📐 Dual-Vector Subspace Projection (Topic vs Affect)</div>
          <div class="wb-radar-row">
            <span>Topic Subspace Energy:</span>
            <span class="wb-radar-val">${topicPct}%</span>
          </div>
          <div class="wb-radar-bar-wrap"><div class="wb-radar-bar" style="width: ${topicPct}%; background: var(--wb-neon-cyan);"></div></div>
          <div class="wb-radar-row">
            <span>Affect Hostility Ratio:</span>
            <span class="wb-radar-val">${affectPct}%</span>
          </div>
          <div class="wb-radar-bar-wrap"><div class="wb-radar-bar" style="width: ${affectPct}%; background: var(--wb-neon-red);"></div></div>
          <div class="wb-radar-note">Direct slur energy: <b>${slurEnergy}</b> ${meta.false_alarm_damped ? "• Topical debate protected" : ""}</div>
        </div>
      `;
    }

    // 2. Multimodal Joint Disparity (Malicious Subtlety)
    if (result.cross_modal_flagged) {
      detailsHtml += `
        <div class="wb-radar-card" style="border-color: var(--wb-neon-pink);">
          <div class="wb-radar-title" style="color: var(--wb-neon-pink);">⚡ Malicious Subtlety (Cross-Modal Disparity)</div>
          <div class="wb-radar-row">
            <span>Disparity Tension:</span>
            <span class="wb-radar-val">${result.cross_modal_disparity}</span>
          </div>
          <div class="wb-radar-note">Innocuous textual surface clashing with antagonistic visual context.</div>
        </div>
      `;
    }

    // 3. Linear Feature Attribution Tokens
    if (result.explanation && result.explanation.length > 0) {
      const termsList = result.explanation
        .map((e) => `<span class="wb-term-tag">"${escapeHtml(e.term)}" (+${e.contribution})</span>`)
        .join(" ");
      detailsHtml += `<div class="wb-explain-desc">Model linear attribution signals:</div>${termsList}`;
    }

    // 4. On-Device IndexedDB Shifts
    if (result.on_device_terms && result.on_device_terms.length > 0) {
      const dbTerms = result.on_device_terms
        .map((t) => `<span class="wb-term-tag" style="border-color: var(--wb-neon-green); color: var(--wb-neon-green);">"${escapeHtml(t.term)}" (${t.weight > 0 ? "+" : ""}${t.weight})</span>`)
        .join(" ");
      detailsHtml += `<div class="wb-explain-desc" style="margin-top: 8px;">On-Device IndexedDB Shifts:</div>${dbTerms}`;
    }

    expContent.innerHTML = detailsHtml || `<div class="wb-explain-desc">Moderation score: ${result.toxicity_score}</div>`;

    expToggle.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isShown = expContent.style.display === "block";
      expContent.style.display = isShown ? "none" : "block";
      expToggle.textContent = isShown ? "ℹ Why was this flagged? (Model Radar)" : "▲ Hide explanation";
    });

    expContainer.appendChild(expToggle);
    expContainer.appendChild(expContent);
    overlay.appendChild(expContainer);

    // Button Group
    const btnGroup = document.createElement("div");
    btnGroup.className = "wb-btn-group";

    const neutBtn = document.createElement("button");
    neutBtn.type = "button";
    neutBtn.className = "wb-neutralize-btn";
    neutBtn.textContent = "🌿 Ambient Neutralize (Calm Read)";
    btnGroup.appendChild(neutBtn);

    const unhideBtn = document.createElement("button");
    unhideBtn.type = "button";
    unhideBtn.className = "wb-unhide-btn";
    unhideBtn.textContent = "Unhide Content";
    btnGroup.appendChild(unhideBtn);

    overlay.appendChild(btnGroup);

    return { overlay, unhideBtn, neutBtn, neutContainer };
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

    const { overlay, unhideBtn, neutBtn, neutContainer } = buildOverlay(result, postText);
    wrapper.appendChild(overlay);

    let revealed = false;
    let neutralized = false;

    // Ambient Neutralize Click (Calm Read)
    neutBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      neutralized = !neutralized;
      neutContainer.style.display = neutralized ? "block" : "none";
      neutBtn.textContent = neutralized ? "✕ Hide Calm Summary" : "🌿 Ambient Neutralize (Calm Read)";
      neutBtn.classList.toggle("wb-btn-active", neutralized);
    });

    // Auto-neutralize if user enabled it in popup settings
    if (settings.autoNeutralizeEnabled && (result.is_toxic || result.cross_modal_flagged)) {
      neutralized = true;
      neutContainer.style.display = "block";
      neutBtn.textContent = "✕ Hide Calm Summary";
      neutBtn.classList.add("wb-btn-active");
    }

    // Unhide Click
    unhideBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      revealed = !revealed;
      node.classList.toggle("wb-target-blurred", !revealed);
      overlay.classList.toggle("wb-hidden", revealed);
      unhideBtn.textContent = revealed ? "Hide Again" : "Unhide Content";

      // Trigger personal adaptation update (both on-device IndexedDB & backend SGD)
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
        } else if (added.querySelectorAll) {
          const children = added.querySelectorAll(selector);
          children.forEach((c) => processNode(c));
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
