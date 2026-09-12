/**
 * background.js
 * --------------
 * MV3 service worker. Proxies backend calls on behalf of content.js.
 *
 * Why this exists: content.js runs injected into the actual web page
 * (https://instagram.com, https://twitter.com, etc). Chrome's Private
 * Network Access policy blocks a secure public page from directly fetching
 * a private/localhost address — so `fetch("http://localhost:8000/...")`
 * called from content.js fails with "TypeError: Failed to fetch", even
 * though the backend is running and popup.js (a chrome-extension:// page,
 * not subject to that restriction) can reach it fine.
 *
 * The fix: content.js sends a chrome.runtime message to this background
 * worker, which performs the actual fetch and relays the result back.
 * Background workers aren't page contexts, so they aren't blocked the
 * same way.
 */

const API_BASE = "http://localhost:8000";

async function forwardToBackend(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} HTTP ${res.status}`);
  return await res.json();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.type) return false;

  if (message.type === "wb-analyze-post") {
    forwardToBackend("/analyze-post", message.payload)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async response
  }

  if (message.type === "wb-check-draft") {
    forwardToBackend("/check-draft", message.payload)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});
