/**
 * popup.js
 * ---------
 * Wires up the popup dashboard:
 *   - Sensitivity slider, master/NSFW/draft-check toggles -> chrome.storage.local
 *   - Trigger-topic tag input -> chrome.storage.local (read by content.js)
 *   - Mood & Wellbeing dashboard -> reads session stats from chrome.storage.local
 *     (written by content.js) and calls the backend's /calculate-mood-impact
 */

const API_BASE = "http://localhost:8000";

const el = {
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdValue: document.getElementById("thresholdValue"),
  enabledToggle: document.getElementById("enabledToggle"),
  nsfwToggle: document.getElementById("nsfwToggle"),
  draftToggle: document.getElementById("draftToggle"),
  triggerTags: document.getElementById("triggerTags"),
  triggerInput: document.getElementById("triggerInput"),
  statTime: document.getElementById("statTime"),
  statBlocked: document.getElementById("statBlocked"),
  moodScoreValue: document.getElementById("moodScoreValue"),
  moodSummary: document.getElementById("moodSummary"),
  refreshMoodBtn: document.getElementById("refreshMoodBtn"),
};

const DEFAULTS = {
  toxicityThreshold: 0.5,
  userTriggers: [],
  nsfwFilteringEnabled: true,
  draftCheckEnabled: true,
  extensionEnabled: true,
};

// ---------------------------------------------------------------------
// Settings: load + persist
// ---------------------------------------------------------------------
function loadSettings() {
  chrome.storage.local.get(Object.keys(DEFAULTS), (stored) => {
    const settings = { ...DEFAULTS, ...stored };

    el.thresholdSlider.value = settings.toxicityThreshold;
    el.thresholdValue.textContent = Number(settings.toxicityThreshold).toFixed(2);
    el.enabledToggle.checked = settings.extensionEnabled;
    el.nsfwToggle.checked = settings.nsfwFilteringEnabled;
    el.draftToggle.checked = settings.draftCheckEnabled;

    renderTags(settings.userTriggers);
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

// ---------------------------------------------------------------------
// Backend health check
// ---------------------------------------------------------------------
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    el.statusDot.className = "status-dot online";
    el.statusText.textContent = data.using_fallback_toxicity
      ? "Online (Detoxify fallback model)"
      : "Online (self-trained model active)";
  } catch (err) {
    el.statusDot.className = "status-dot offline";
    el.statusText.textContent = "Backend offline — start main.py on :8000";
  }
}

// ---------------------------------------------------------------------
// Mood & Wellbeing dashboard
// ---------------------------------------------------------------------
function getSessionStats() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["sessionStats", "sessionStartTimestamp"], (data) => {
      resolve({
        stats: data.sessionStats || { toxicBlocked: 0, nsfwBlocked: 0, triggerBlocked: 0 },
        startedAt: data.sessionStartTimestamp || Date.now(),
      });
    });
  });
}

async function refreshMoodDashboard() {
  const { stats, startedAt } = await getSessionStats();
  const minutes = Math.max(0, (Date.now() - startedAt) / 60000);
  const totalBlocked = stats.toxicBlocked + stats.nsfwBlocked + stats.triggerBlocked;

  el.statTime.textContent = `${minutes.toFixed(0)}m`;
  el.statBlocked.textContent = totalBlocked;

  try {
    const res = await fetch(`${API_BASE}/calculate-mood-impact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_duration_minutes: minutes,
        toxic_blocked_count: stats.toxicBlocked,
        nsfw_blocked_count: stats.nsfwBlocked,
        trigger_blocked_count: stats.triggerBlocked,
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    el.moodScoreValue.textContent = `${data.mood_preservation_score.toFixed(0)}%`;
    el.moodSummary.textContent = data.summary;
  } catch (err) {
    el.moodScoreValue.textContent = "—";
    el.moodSummary.textContent = "Couldn't reach the backend for a mood score.";
  }
}

el.refreshMoodBtn.addEventListener("click", refreshMoodDashboard);

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
loadSettings();
checkBackendHealth();
refreshMoodDashboard();
