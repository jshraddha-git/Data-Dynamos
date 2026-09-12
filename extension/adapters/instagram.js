/**
 * instagram.js
 * ------------
 * Platform adapter for Instagram (instagram.com).
 * Targets feed articles, modal dialog posts, reels, and comment threads.
 */

class InstagramAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("Instagram", ["instagram.com"]);
  }

  getPostSelector() {
    return "article, div[role='presentation'] article, div._aagv";
  }

  getComposeSelector() {
    return "textarea[aria-label*='Add a comment'], div[role='textbox'][contenteditable='true']";
  }

  extractText(node) {
    // Caption is usually in h1 or span._ap3a or div._a9zs
    const captionEl = node.querySelector("h1, span._ap3a, div._a9zs, div[role='button'] span");
    if (captionEl) return (captionEl.innerText || "").trim().slice(0, 2500);
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    const mediaEls = node.querySelectorAll("div._aagv img, img.x5yr21d");
    mediaEls.forEach((img) => {
      const src = img.src || img.getAttribute("srcset")?.split(" ")[0];
      if (src && src.startsWith("http")) images.push(src);
    });
    if (images.length > 0) return images.slice(0, 4);
    return super.extractImages(node);
  }
}

window.InstagramAdapter = InstagramAdapter;
