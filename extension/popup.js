/**
 * popup.js
 * ---------
 * Wires up the extension popup settings, live mood score, and server connection.
 */

const DEFAULT_API_BASE = "http://localhost:8000";

const el = {
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdValue: document.getElementById("thresholdValue"),
  enabledToggle: document.getElementById("enabledToggle"),
  nsfwToggle: document.getElementById("nsfwToggle"),
  draftToggle: document.getElementById("draftToggle"),
  adaptiveToggle: document.getElementById("adaptiveToggle"),
  adaptiveStatus: document.getElementById("adaptiveStatus"),
  triggerTags: document.getElementById("triggerTags"),
  triggerInput: document.getElementById("triggerInput"),
  statTime: document.getElementById("statTime"),
  statBlocked: document.getElementById("statBlocked"),
  moodScoreValue: document.getElementById("moodScoreValue"),
  moodSummary: document.getElementById("moodSummary"),
  refreshMoodBtn: document.getElementById("refreshMoodBtn"),
  serverSettingsToggle: document.getElementById("serverSettingsToggle"),
  serverSettingsBody: document.getElementById("serverSettingsBody"),
  apiBaseInput: document.getElementById("apiBaseInput"),
  apiKeyInput: document.getElementById("apiKeyInput"),
  saveServerBtn: document.getElementById("saveServerBtn"),
  statLearnedBias: document.getElementById("statLearnedBias"),
  statActiveTerms: document.getElementById("statActiveTerms"),
  toleratedTags: document.getElementById("toleratedTags"),
  refreshPersonalBtn: document.getElementById("refreshPersonalBtn"),
  resetPersonalBtn: document.getElementById("resetPersonalBtn"),
};

const DEFAULTS = {
  toxicityThreshold: 0.5,
  userTriggers: ["spoilers", "layoffs"],
  nsfwFilteringEnabled: true,
  draftCheckEnabled: true,
  extensionEnabled: true,
  adaptiveSensitivityEnabled: true,
  learnedSensitivityAdjustment: 0.0,
  apiBaseUrl: DEFAULT_API_BASE,
  apiKey: "",
};

// ---------------------------------------------------------------------
// Load & Render Settings
// ---------------------------------------------------------------------
function loadSettings() {
  chrome.storage.local.get(Object.keys(DEFAULTS), (stored) => {
    const settings = { ...DEFAULTS, ...stored };

    el.thresholdSlider.value = settings.toxicityThreshold;
    el.thresholdValue.textContent = Number(settings.toxicityThreshold).toFixed(2);
    el.enabledToggle.checked = settings.extensionEnabled;
    el.nsfwToggle.checked = settings.nsfwFilteringEnabled;
    el.draftToggle.checked = settings.draftCheckEnabled;
    el.adaptiveToggle.checked = settings.adaptiveSensitivityEnabled;

    // Display adaptive adjustment status
    const adj = settings.learnedSensitivityAdjustment || 0.0;
    const sign = adj >= 0 ? "+" : "";
    el.adaptiveStatus.textContent = settings.adaptiveSensitivityEnabled
      ? `Shift: ${sign}${adj.toFixed(2)}`
      : "Disabled";

    el.apiBaseInput.value = settings.apiBaseUrl;
    el.apiKeyInput.value = settings.apiKey;

    renderTags(settings.userTriggers);
    checkBackendHealth(settings.apiBaseUrl, settings.apiKey);
  });
}

function renderTags(triggers) {
  el.triggerTags.innerHTML = "";
  triggers.forEach((topic, idx) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.innerHTML = `${escapeHtml(topic)} <button data-idx="${idx}" title="Remove">✕</button>`;
    el.triggerTags.appendChild(tag);
  });

  el.triggerTags.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.idx);
      chrome.storage.local.get(["userTriggers"], ({ userTriggers = [] }) => {
        const updated = userTriggers.filter((_, i) => i !== idx);
        chrome.storage.local.set({ userTriggers: updated }, () => renderTags(updated));
      });
    });
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------
el.thresholdSlider.addEventListener("input", () => {
  const value = parseFloat(el.thresholdSlider.value);
  el.thresholdValue.textContent = value.toFixed(2);
  chrome.storage.local.set({ toxicityThreshold: value });
});

el.enabledToggle.addEventListener("change", () => {
  chrome.storage.local.set({ extensionEnabled: el.enabledToggle.checked });
});

el.nsfwToggle.addEventListener("change", () => {
  chrome.storage.local.set({ nsfwFilteringEnabled: el.nsfwToggle.checked });
});

el.draftToggle.addEventListener("change", () => {
  chrome.storage.local.set({ draftCheckEnabled: el.draftToggle.checked });
});

el.adaptiveToggle.addEventListener("change", () => {
  chrome.storage.local.set({ adaptiveSensitivityEnabled: el.adaptiveToggle.checked });
  el.adaptiveStatus.textContent = el.adaptiveToggle.checked ? "Active" : "Disabled";
});

el.triggerInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const topic = el.triggerInput.value.trim();
  if (!topic) return;
  chrome.storage.local.get(["userTriggers"], ({ userTriggers = [] }) => {
    if (userTriggers.includes(topic)) {
      el.triggerInput.value = "";
      return;
    }
    const updated = [...userTriggers, topic];
    chrome.storage.local.set({ userTriggers: updated }, () => {
      renderTags(updated);
      el.triggerInput.value = "";
    });
  });
});

// Collapsible Server Settings
el.serverSettingsToggle.addEventListener("click", () => {
  const isHidden = el.serverSettingsBody.style.display === "none";
  el.serverSettingsBody.style.display = isHidden ? "block" : "none";
  el.serverSettingsToggle.textContent = isHidden ? "⚙ Server Connection ▴" : "⚙ Server Connection ▾";
});

el.saveServerBtn.addEventListener("click", () => {
  const apiBaseUrl = el.apiBaseInput.value.trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
  const apiKey = el.apiKeyInput.value.trim();
  chrome.storage.local.set({ apiBaseUrl, apiKey }, () => {
    checkBackendHealth(apiBaseUrl, apiKey);
    refreshMoodDashboard();
  });
});

// ---------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------
async function checkBackendHealth(apiBase = DEFAULT_API_BASE, apiKey = "") {
  try {
    const headers = {};
    if (apiKey) headers["X-API-Key"] = apiKey;

    const res = await fetch(`${apiBase}/health`, { method: "GET", headers });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    el.statusDot.className = "status-dot online";
    const models = [];
    if (data.primary_toxicity_model_loaded) models.push("Toxicity");
    if (data.self_trained_lsa_model_loaded) models.push("LSA");
    el.statusText.textContent = `Online (${models.join("+")} Self-Trained)`;
  } catch (err) {
    el.statusDot.className = "status-dot offline";
    el.statusText.textContent = "Backend offline — check server URL";
  }
}

// ---------------------------------------------------------------------
// Mood Dashboard
// ---------------------------------------------------------------------
function getSessionStats() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["sessionStats", "sessionStartTimestamp"], (data) => {
      resolve({
        stats: data.sessionStats || { toxicBlocked: 0, nsfwBlocked: 0, triggerBlocked: 0, memeBlocked: 0 },
        startedAt: data.sessionStartTimestamp || Date.now(),
      });
    });
  });
}

async function refreshMoodDashboard() {
  const { stats, startedAt } = await getSessionStats();
  const minutes = Math.max(0, (Date.now() - startedAt) / 60000);
  const totalBlocked =
    (stats.toxicBlocked || 0) +
    (stats.nsfwBlocked || 0) +
    (stats.triggerBlocked || 0) +
    (stats.memeBlocked || 0);

  el.statTime.textContent = `${minutes.toFixed(0)}m`;
  el.statBlocked.textContent = totalBlocked;

  chrome.storage.local.get(["apiBaseUrl", "apiKey"], async (cfg) => {
    const apiBase = (cfg.apiBaseUrl || DEFAULT_API_BASE).replace(/\/+$/, "");
    const headers = { "Content-Type": "application/json" };
    if (cfg.apiKey) headers["X-API-Key"] = cfg.apiKey;

    try {
      const res = await fetch(`${apiBase}/calculate-mood-impact`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          session_duration_minutes: minutes,
          toxic_blocked_count: stats.toxicBlocked || 0,
          nsfw_blocked_count: stats.nsfwBlocked || 0,
          trigger_blocked_count: (stats.triggerBlocked || 0) + (stats.memeBlocked || 0),
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      el.moodScoreValue.textContent = `${data.mood_preservation_score.toFixed(0)}%`;
      el.moodSummary.textContent = data.summary;
    } catch (err) {
      el.moodScoreValue.textContent = "—";
      el.moodSummary.textContent = "Could not reach backend for mood score.";
    }
  });
}

el.refreshMoodBtn.addEventListener("click", refreshMoodDashboard);

// ---------------------------------------------------------------------
// Adaptive Personalization Stats & Control (USP)
// ---------------------------------------------------------------------
async function refreshPersonalizationStats() {
  chrome.storage.local.get(["apiBaseUrl", "apiKey"], async (cfg) => {
    const apiBase = (cfg.apiBaseUrl || DEFAULT_API_BASE).replace(/\/+$/, "");
    const headers = {};
    if (cfg.apiKey) headers["X-API-Key"] = cfg.apiKey;

    try {
      const res = await fetch(`${apiBase}/personalize-stats/default`, { method: "GET", headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (el.statLearnedBias) {
        const bias = data.learned_bias || 0.0;
        const sign = bias > 0 ? "+" : "";
        el.statLearnedBias.textContent = `${sign}${bias.toFixed(3)}`;
      }
      if (el.statActiveTerms) {
        el.statActiveTerms.textContent = data.active_feature_shifts || 0;
      }

      if (el.toleratedTags) {
        el.toleratedTags.innerHTML = "";
        const terms = data.top_tolerated || [];
        if (terms.length === 0) {
          el.toleratedTags.innerHTML = '<span style="color: var(--text-dim); font-size: 11px;">(None yet — unhide posts to adapt)</span>';
        } else {
          terms.forEach((item) => {
            const span = document.createElement("span");
            span.className = "tag";
            span.style.cssText = "background: rgba(55, 242, 161, 0.15); border: 1px solid var(--green); color: var(--green); padding: 2px 8px; font-size: 11px; border-radius: 999px;";
            span.textContent = `"${item.term}" (${item.shift.toFixed(3)})`;
            el.toleratedTags.appendChild(span);
          });
        }
      }
    } catch (err) {
      if (el.statLearnedBias) el.statLearnedBias.textContent = "—";
      if (el.statActiveTerms) el.statActiveTerms.textContent = "—";
    }
  });
}

async function resetPersonalizationModel() {
  chrome.storage.local.get(["apiBaseUrl", "apiKey"], async (cfg) => {
    const apiBase = (cfg.apiBaseUrl || DEFAULT_API_BASE).replace(/\/+$/, "");
    const headers = {};
    if (cfg.apiKey) headers["X-API-Key"] = cfg.apiKey;

    try {
      await fetch(`${apiBase}/personalize-reset/default`, { method: "POST", headers });
      chrome.storage.local.set({ learnedSensitivityAdjustment: 0.0, unhideCount: 0 });
      refreshPersonalizationStats();
    } catch (err) {
      console.warn("Reset personalization model failed:", err);
    }
  });
}

if (el.refreshPersonalBtn) el.refreshPersonalBtn.addEventListener("click", refreshPersonalizationStats);
if (el.resetPersonalBtn) el.resetPersonalBtn.addEventListener("click", resetPersonalizationModel);

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
loadSettings();
refreshMoodDashboard();
refreshPersonalizationStats();
