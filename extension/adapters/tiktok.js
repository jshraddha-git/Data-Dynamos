/**
 * tiktok.js
 * ---------
 * Platform adapter for TikTok (tiktok.com).
 * Targets video feed cards, recommendation items, and comment lists.
 */

class TikTokAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("TikTok", ["tiktok.com"]);
  }

  getPostSelector() {
    return "div[data-e2e='recommend-list-item-container'], div[data-e2e='comment-item'], div.css-1qjw46g-DivItemContainerV2";
  }

  getComposeSelector() {
    return "div[data-e2e='comment-input'] [contenteditable='true'], div.public-DraftEditor-content";
  }

  extractText(node) {
    // Caption or comment text
    const descEl = node.querySelector("[data-e2e='browse-video-desc'], [data-e2e='comment-level-1'], p[data-e2e='video-desc']");
    if (descEl) return (descEl.innerText || "").trim().slice(0, 2500);
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractMedia(node) {
    const media = [];
    const videoEl = node.querySelector("video");
    if (videoEl) {
      if (videoEl.poster) media.push(videoEl.poster);
      const frame = this.sampleVideoFrame(videoEl);
      if (frame) media.push(frame);
    }
    const imgEls = node.querySelectorAll("img[src*='tiktokcdn']");
    imgEls.forEach((img) => {
      if (img.src && !img.src.includes("avatar")) media.push(img.src);
    });
    return media.slice(0, 4);
  }
}

window.TikTokAdapter = TikTokAdapter;
