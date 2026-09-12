/**
 * base_adapter.js
 * ----------------
 * Base class for platform adapters. Defines standard interface for
 * locating post nodes, extracting text, sampling media (images + video frames),
 * and watching compose boxes across diverse social media platforms.
 */

class BasePlatformAdapter {
  constructor(name, domains = []) {
    this.name = name;
    this.domains = domains;
    this._canvas = null;
  }

  matches(hostname) {
    return this.domains.some((d) => hostname === d || hostname.endsWith(`.${d}`));
  }

  /** CSS selector identifying individual post or comment containers */
  getPostSelector() {
    return "article, [role='article'], .post, .comment";
  }

  /** CSS selector identifying text inputs / compose boxes */
  getComposeSelector() {
    return "textarea, [contenteditable='true'], input[type='text'], div[role='textbox']";
  }

  /** Extracts text content from a post node */
  extractText(node) {
    return (node.innerText || "").trim().slice(0, 2500);
  }

  /** Extracts static image URLs from a post node */
  extractImages(node) {
    return Array.from(node.querySelectorAll("img"))
      .map((img) => img.src || img.getAttribute("data-src"))
      .filter((src) => src && src.startsWith("http") && !src.includes("/emoji/") && !src.includes("/avatar"))
      .slice(0, 4);
  }

  /**
   * Samples a frame from an active HTML5 video element using an offscreen canvas.
   * Returns a Base64 data URL string (data:image/jpeg;base64,...) or null.
   */
  sampleVideoFrame(videoEl) {
    if (!videoEl || videoEl.readyState < 2) return null; // HAVE_CURRENT_DATA
    try {
      if (!this._canvas) {
        this._canvas = document.createElement("canvas");
      }
      const width = videoEl.videoWidth || 320;
      const height = videoEl.videoHeight || 240;
      // Cap dimensions to keep payload lightweight (<100KB)
      const scale = Math.min(1.0, 320 / Math.max(width, height));
      this._canvas.width = Math.round(width * scale);
      this._canvas.height = Math.round(height * scale);
      const ctx = this._canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0, this._canvas.width, this._canvas.height);
      return this._canvas.toDataURL("image/jpeg", 0.7);
    } catch (err) {
      // Catch possible canvas CORS taint if video is cross-origin without crossorigin attribute
      return null;
    }
  }

  /** Extracts video thumbnails and samples active video frames */
  extractMedia(node) {
    const images = this.extractImages(node);
    const mediaUrls = [...images];

    // Check for video elements
    const videos = Array.from(node.querySelectorAll("video"));
    for (const v of videos) {
      if (v.poster && v.poster.startsWith("http")) {
        mediaUrls.push(v.poster);
      } else {
        const frameData = this.sampleVideoFrame(v);
        if (frameData) mediaUrls.push(frameData);
      }
    }
    return mediaUrls.slice(0, 5);
  }

  /** Extracts structured post data from a post node */
  extractPostData(node) {
    const text = this.extractText(node);
    const media = this.extractMedia(node);
    return {
      text,
      media,
      platform: this.name,
      id: node.id || node.getAttribute("data-id") || null,
    };
  }

  /** Extracts draft text from a compose input */
  getDraftText(inputEl) {
    if (inputEl.value !== undefined) return inputEl.value;
    return inputEl.innerText || "";
  }
}

// Attach to window so all content scripts can access in browser environment
window.BasePlatformAdapter = BasePlatformAdapter;
