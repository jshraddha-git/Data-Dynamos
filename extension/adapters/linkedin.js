/**
 * linkedin.js
 * -----------
 * Platform adapter for LinkedIn (linkedin.com).
 * Targets feed updates, sponsored posts, articles, and comments.
 */

class LinkedInAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("LinkedIn", ["linkedin.com"]);
  }

  getPostSelector() {
    return "div.feed-shared-update-v2, div.comments-comment-item, div.feed-shared-update-v2__control-menu-container";
  }

  getComposeSelector() {
    return "div.ql-editor[contenteditable='true'], div.comments-comment-box__editor";
  }

  extractText(node) {
    const textEl = node.querySelector("span.break-words, div.feed-shared-update-v2__description, div.comments-comment-item__main-content");
    if (textEl) return (textEl.innerText || "").trim().slice(0, 2500);
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    const mediaEls = node.querySelectorAll("img.feed-shared-image__image, img.ivm-view-attr__img--centered");
    mediaEls.forEach((img) => {
      if (img.src && img.src.startsWith("http") && !img.src.includes("ghost")) {
        images.push(img.src);
      }
    });
    if (images.length > 0) return images.slice(0, 4);
    return super.extractImages(node);
  }
}

window.LinkedInAdapter = LinkedInAdapter;
