/**
 * popup.js
 * --------
 * Wires up the popup dashboard:
 *   - Sensitivity threshold slider -> chrome.storage.local.toxicityThreshold
 *   - Trigger topic tags -> chrome.storage.local.userTriggers
 *   - NSFW / draft-check toggles -> chrome.storage.local.{nsfwFilterEnabled,draftCheckEnabled}
 *   - Mood & Wellbeing Dashboard -> reads session counters from storage,
 *     POSTs them to /calculate-mood-impact, and renders the result.
 */

const API_BASE = "http://localhost:8000";

const DEFAULT_SETTINGS = {
  userTriggers: [],
  toxicityThreshold: 0.5,
  nsfwFilterEnabled: true,
  draftCheckEnabled: true,
};

const els = {
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdValue: document.getElementById("thresholdValue"),
  tagInput: document.getElementById("tagInput"),
  addTagBtn: document.getElementById("addTagBtn"),
  tagList: document.getElementById("tagList"),
  nsfwToggle: document.getElementById("nsfwToggle"),
  draftToggle: document.getElementById("draftToggle"),
  scrollTimeValue: document.getElementById("scrollTimeValue"),
  totalBlockedValue: document.getElementById("totalBlockedValue"),
  moodScoreValue: document.getElementById("moodScoreValue"),
  moodSummary: document.getElementById("moodSummary"),
  resetSessionBtn: document.getElementById("resetSessionBtn"),
  backendStatus: document.getElementById("backendStatus"),
};

let currentTriggers = [];

// ---------------------------------------------------------------------
// Settings: load + render
// ---------------------------------------------------------------------
function loadSettingsIntoUI() {
  chrome.storage.local.get(DEFAULT_SETTINGS, (settings) => {
    els.thresholdSlider.value = settings.toxicityThreshold;
    els.thresholdValue.textContent = Number(settings.toxicityThreshold).toFixed(2);
    els.nsfwToggle.checked = !!settings.nsfwFilterEnabled;
    els.draftToggle.checked = !!settings.draftCheckEnabled;
    currentTriggers = Array.isArray(settings.userTriggers) ? settings.userTriggers : [];
    renderTagList();
  });
}

function renderTagList() {
  els.tagList.innerHTML = "";
  currentTriggers.forEach((tag, idx) => {
    const chip = document.createElement("div");
    chip.className = "tag-chip";

    const label = document.createElement("span");
    label.textContent = tag;

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "✕";
    removeBtn.addEventListener("click", () => {
      currentTriggers.splice(idx, 1);
      persistTriggers();
      renderTagList();
    });

    chip.appendChild(label);
    chip.appendChild(removeBtn);
    els.tagList.appendChild(chip);
  });
}

function persistTriggers() {
  chrome.storage.local.set({ userTriggers: currentTriggers });
}

// ---------------------------------------------------------------------
// Event wiring: threshold, tags, toggles
// ---------------------------------------------------------------------
els.thresholdSlider.addEventListener("input", () => {
  const val = parseFloat(els.thresholdSlider.value);
  els.thresholdValue.textContent = val.toFixed(2);
  chrome.storage.local.set({ toxicityThreshold: val });
});

els.addTagBtn.addEventListener("click", addTagFromInput);
els.tagInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    addTagFromInput();
  }
});

function addTagFromInput() {
  const raw = els.tagInput.value.trim();
  if (!raw) return;

  // Support comma-separated bulk entry, e.g. "spoilers, politics, crypto".
  const newTags = raw
    .split(",")
    .map((t) => t.trim().toLowerCase())
    .filter((t) => t.length > 0 && !currentTriggers.includes(t));

  currentTriggers = currentTriggers.concat(newTags);
  persistTriggers();
  renderTagList();
  els.tagInput.value = "";
}

els.nsfwToggle.addEventListener("change", () => {
  chrome.storage.local.set({ nsfwFilterEnabled: els.nsfwToggle.checked });
});

els.draftToggle.addEventListener("change", () => {
  chrome.storage.local.set({ draftCheckEnabled: els.draftToggle.checked });
});

// ---------------------------------------------------------------------
// Mood & Wellbeing Dashboard
// ---------------------------------------------------------------------
async function refreshMoodDashboard() {
  chrome.storage.local.get(
    [
      "sessionStart",
      "toxicBlockedCount",
      "nsfwBlockedCount",
      "triggerBlockedCount",
    ],
    async (data) => {
      const sessionStart = data.sessionStart || Date.now();
      const toxicBlockedCount = data.toxicBlockedCount || 0;
      const nsfwBlockedCount = data.nsfwBlockedCount || 0;
      const triggerBlockedCount = data.triggerBlockedCount || 0;

      const minutes = Math.max(0, (Date.now() - sessionStart) / 60000);
      const totalBlocked = toxicBlockedCount + nsfwBlockedCount + triggerBlockedCount;

      els.scrollTimeValue.textContent = `${Math.round(minutes)} Mins`;
      els.totalBlockedValue.textContent = String(totalBlocked);

      try {
        const res = await fetch(`${API_BASE}/calculate-mood-impact`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_duration_minutes: minutes,
            toxic_blocked_count: toxicBlockedCount,
            nsfw_blocked_count: nsfwBlockedCount,
            trigger_blocked_count: triggerBlockedCount,
          }),
        });

        if (!res.ok) throw new Error(`Backend returned ${res.status}`);
        const result = await res.json();

        els.moodScoreValue.textContent = `${Math.round(result.mood_preservation_score)}%`;
        els.moodSummary.textContent = result.summary;
        els.backendStatus.textContent = "";
      } catch (err) {
        els.moodScoreValue.textContent = "--%";
        els.moodSummary.textContent = "";
        els.backendStatus.textContent =
          "Backend unreachable — start the FastAPI server on localhost:8000.";
      }
    }
  );
}

els.resetSessionBtn.addEventListener("click", () => {
  chrome.storage.local.set(
    {
      sessionStart: Date.now(),
      toxicBlockedCount: 0,
      nsfwBlockedCount: 0,
      triggerBlockedCount: 0,
    },
    refreshMoodDashboard
  );
});

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
loadSettingsIntoUI();
refreshMoodDashboard();
setInterval(refreshMoodDashboard, 5000);
