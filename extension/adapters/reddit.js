/**
 * reddit.js
 * ---------
 * Platform adapter for Reddit (reddit.com, old.reddit.com).
 * Targets modern shreddit Web Components (shreddit-post, shreddit-comment)
 * and legacy Reddit DOM elements (.Comment, .entry).
 */

class RedditAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("Reddit", ["reddit.com"]);
  }

  getPostSelector() {
    return "shreddit-post, shreddit-comment, div[data-testid='post-container'], .Comment, .entry";
  }

  getComposeSelector() {
    return "shreddit-composer, div[role='textbox'][contenteditable='true'], textarea[name='text']";
  }

  extractText(node) {
    // 1. Shreddit post title + body
    const titleEl = node.querySelector("[slot='title'], h1, h2, a[data-testid='post-title']");
    const bodyEl = node.querySelector("[slot='text-body'], div[data-testid='post-content'], .usertext-body");
    const title = titleEl ? titleEl.innerText.trim() : "";
    const body = bodyEl ? bodyEl.innerText.trim() : "";

    if (title || body) {
      return `${title}\n${body}`.trim().slice(0, 2500);
    }
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    // Shreddit media lightbox or image gallery
    const imgEls = node.querySelectorAll("img[src*='redd.it'], img[src*='redditmedia'], shreddit-player img");
    imgEls.forEach((img) => {
      const src = img.src || img.getAttribute("data-src");
      if (src && src.startsWith("http")) images.push(src);
    });
    if (images.length > 0) return images.slice(0, 4);
    return super.extractImages(node);
  }
}

window.RedditAdapter = RedditAdapter;
