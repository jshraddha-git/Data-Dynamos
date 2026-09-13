/**
 * indexeddb_vector_engine.js
 * ---------------------------
 * Model Adaptation USP: On-Device Vector Shifts via IndexedDB.
 * Fine-tunes moderation sensitivity locally inside the browser's persistent IndexedDB.
 *
 * Privacy & Architecture:
 *   - Personal adaptation weights are stored in the extension's local origin.
 *   - Zero network leakage: personal tolerance never needs to leave the device.
 *   - Performs sparse Online SGD updates directly inside IndexedDB transactions.
 */

(() => {
  const DB_NAME = "AIWellbeingDB";
  const DB_VERSION = 1;
  const STORE_SHIFTS = "user_vector_shifts";
  const STORE_EVENTS = "adaptation_events";

  const STOP_WORDS = new Set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "get", "got", "look", "make", "know", "think", "see", "come", "want", "give"
  ]);

  let dbPromise = null;

  function openDB() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = (evt) => {
        const db = evt.target.result;
        if (!db.objectStoreNames.contains(STORE_SHIFTS)) {
          db.createObjectStore(STORE_SHIFTS, { keyPath: "term" });
        }
        if (!db.objectStoreNames.contains(STORE_EVENTS)) {
          db.createObjectStore(STORE_EVENTS, { keyPath: "id", autoIncrement: true });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return dbPromise;
  }

  function extractInformativeTokens(text) {
    if (!text) return [];
    const rawWords = text.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
    return [...new Set(rawWords.filter((w) => !STOP_WORDS.has(w)))].slice(0, 15);
  }

  /**
   * Reads all learned vector shifts from IndexedDB.
   */
  async function getAllShifts() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SHIFTS, "readonly");
      const store = tx.objectStore(STORE_SHIFTS);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  /**
   * Computes the on-device vector shift for a given post text.
   */
  async function computeLocalShift(text) {
    const tokens = extractInformativeTokens(text);
    if (tokens.length === 0) return { shift: 0.0, matchedTerms: [] };

    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_SHIFTS, "readonly");
      const store = tx.objectStore(STORE_SHIFTS);
      let totalShift = 0.0;
      const matched = [];
      let pending = tokens.length;

      tokens.forEach((term) => {
        const req = store.get(term);
        req.onsuccess = () => {
          if (req.result && typeof req.result.weight === "number") {
            totalShift += req.result.weight;
            matched.push({ term: req.result.term, weight: req.result.weight });
          }
          pending -= 1;
          if (pending === 0) {
            // Bound shift to [-0.15, +0.15]
            const clamped = Math.max(-0.15, Math.min(0.15, totalShift));
            resolve({ shift: parseFloat(clamped.toFixed(4)), matchedTerms: matched });
          }
        };
        req.onerror = () => {
          pending -= 1;
          if (pending === 0) {
            resolve({ shift: 0.0, matchedTerms: matched });
          }
        };
      });
    });
  }

  /**
   * Executes an on-device Online SGD step on user feedback (unhide / rehide)
   * and persists updated weights to IndexedDB.
   */
  async function recordFeedbackStep(text, action, baseScore = 0.5) {
    const tokens = extractInformativeTokens(text);
    if (tokens.length === 0) return { updated: 0 };

    const targetY = action === "unhide" ? 0.0 : 1.0;
    const learningRate = 0.035;
    const l2Reg = 0.01;
    const error = baseScore - targetY;

    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction([STORE_SHIFTS, STORE_EVENTS], "readwrite");
      const shiftsStore = tx.objectStore(STORE_SHIFTS);
      const eventsStore = tx.objectStore(STORE_EVENTS);

      let completed = 0;
      tokens.forEach((term) => {
        const getReq = shiftsStore.get(term);
        getReq.onsuccess = () => {
          const curr = (getReq.result && getReq.result.weight) || 0.0;
          let nextW = (1.0 - l2Reg) * curr - learningRate * error;
          nextW = Math.max(-0.06, Math.min(0.06, nextW));

          if (Math.abs(nextW) > 0.005) {
            shiftsStore.put({
              term,
              weight: parseFloat(nextW.toFixed(4)),
              updated_at: Date.now()
            });
          } else {
            shiftsStore.delete(term);
          }
          completed += 1;
          if (completed === tokens.length) {
            eventsStore.add({
              action,
              tokens,
              baseScore,
              timestamp: Date.now()
            });
          }
        };
      });

      tx.oncomplete = () => resolve({ updated: tokens.length, action });
      tx.onerror = () => reject(tx.error);
    });
  }

  /**
   * Resets all stored on-device vector shifts in IndexedDB.
   */
  async function resetShifts() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction([STORE_SHIFTS, STORE_EVENTS], "readwrite");
      tx.objectStore(STORE_SHIFTS).clear();
      tx.objectStore(STORE_EVENTS).clear();
      tx.oncomplete = () => resolve({ status: "cleared" });
      tx.onerror = () => reject(tx.error);
    });
  }

  // Export engine
  const IndexedDBVectorEngine = {
    getAllShifts,
    computeLocalShift,
    recordFeedbackStep,
    resetShifts,
    extractInformativeTokens
  };

  if (typeof window !== "undefined") {
    window.IndexedDBVectorEngine = IndexedDBVectorEngine;
  }
  if (typeof self !== "undefined") {
    self.IndexedDBVectorEngine = IndexedDBVectorEngine;
  }
})();
