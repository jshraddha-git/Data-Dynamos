/**
 * background.js
 * --------------
 * Manifest V3 service worker for AI Wellbeing Buffer.
 *
 * Responsibilities:
 *   1. Proxies all ML inference requests on behalf of content scripts to bypass
 *      browser Private Network Access (PNA) restrictions during local development
 *      and enforce centralized secret isolation in production.
 *   2. Keeps the API key secure inside the extension service worker context rather
 *      than exposing it to host web page DOMs.
 *   3. Coordinates global, cross-tab request queuing so multiple concurrent tabs
 *      (e.g. Twitter + Reddit + Instagram) do not overwhelm the hosted backend.
 *   4. Reads backend URL and API Key dynamically from chrome.storage.local.
 */

const DEFAULT_API_BASE = "http://localhost:8000";

async function getBackendConfig() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["apiBaseUrl", "apiKey"], (stored) => {
      resolve({
        apiBase: (stored.apiBaseUrl || DEFAULT_API_BASE).replace(/\/+$/, ""),
        apiKey: stored.apiKey || "",
      });
    });
  });
}

async function forwardToBackend(path, body = null, method = "POST") {
  const { apiBase, apiKey } = await getBackendConfig();
  const headers = {};
  if (body) {
    headers["Content-Type"] = "application/json";
  }
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  const reqOptions = {
    method,
    headers,
  };
  if (body && method !== "GET") {
    reqOptions.body = JSON.stringify(body);
  }

  const res = await fetch(`${apiBase}${path}`, reqOptions);

  if (res.status === 429) {
    const retryAfter = res.headers.get("Retry-After") || "5";
    throw new Error(`Rate limited by backend. Retry in ${retryAfter}s`);
  }

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`${path} HTTP ${res.status}: ${errText || res.statusText}`);
  }

  return await res.json();
}

// ---------------------------------------------------------------------------
// Centralized Cross-Tab Request Queue
// ---------------------------------------------------------------------------
const MAX_GLOBAL_CONCURRENT = 4;
let activeGlobalRequests = 0;
const globalQueue = [];

function drainGlobalQueue() {
  while (activeGlobalRequests < MAX_GLOBAL_CONCURRENT && globalQueue.length > 0) {
    const { task, resolve, reject } = globalQueue.shift();
    activeGlobalRequests += 1;
    task()
      .then(resolve)
      .catch(reject)
      .finally(() => {
        activeGlobalRequests -= 1;
        drainGlobalQueue();
      });
  }
}

function queueRequest(task) {
  return new Promise((resolve, reject) => {
    globalQueue.push({ task, resolve, reject });
    drainGlobalQueue();
  });
}

// ---------------------------------------------------------------------------
// Message Router
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.type) return false;

  if (message.type === "wb-analyze-post") {
    queueRequest(() => forwardToBackend("/analyze-post", message.payload, "POST"))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // Keep message channel open for async response
  }

  if (message.type === "wb-check-draft") {
    queueRequest(() => forwardToBackend("/check-draft", message.payload, "POST"))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "wb-personalize-feedback") {
    queueRequest(() => forwardToBackend("/personalize-feedback", message.payload, "POST"))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "wb-personalize-stats") {
    const clientId = (message.payload && message.payload.clientId) || "default";
    queueRequest(() => forwardToBackend(`/personalize-stats/${clientId}`, null, "GET"))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "wb-personalize-reset") {
    const clientId = (message.payload && message.payload.clientId) || "default";
    queueRequest(() => forwardToBackend(`/personalize-reset/${clientId}`, null, "POST"))
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  if (message.type === "wb-calculate-mood") {
    forwardToBackend("/calculate-mood-impact", message.payload, "POST")
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});
