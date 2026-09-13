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
    return "shreddit-post, div[data-testid='post-container'], shreddit-comment, .Comment, .entry";
  }

  getComposeSelector() {
    return "shreddit-composer, div[role='textbox'][contenteditable='true'], textarea[name='text']";
  }

  extractText(node) {
    // 1. Check attribute directly on shreddit-post tag (present immediately upon DOM creation)
    let title = (node.getAttribute && node.getAttribute("post-title")) || "";
    if (!title) {
      const titleEl = node.querySelector("[slot='title'], h1, h2, h3, a[data-testid='post-title'], a[slot='full-post-link']");
      if (titleEl) title = titleEl.innerText.trim();
    }

    // 2. Body text from slots or text containers
    const bodyEl = node.querySelector("[slot='text-body'], div[data-testid='post-content'], .usertext-body, [slot='comment'], div[id$='-post-rtjson-content'], [data-click-id='text']");
    const body = bodyEl ? bodyEl.innerText.trim() : "";

    if (title || body) {
      return `${title}\n${body}`.trim().slice(0, 2500);
    }
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    // 1. Shreddit media lightbox, aspect-ratio containers, or image gallery
    const imgEls = node.querySelectorAll("img[src*='redd.it'], img[src*='redditmedia'], img[src*='preview.redd.it'], shreddit-player img, shreddit-aspect-ratio img, img");
    imgEls.forEach((img) => {
      const src = img.src || img.getAttribute("data-src") || img.getAttribute("data-lazy-src");
      if (src && src.startsWith("http") && !src.includes("/avatar") && !src.includes("/emoji") && !src.includes("styles/profile")) {
        images.push(src);
      }
    });

    // 2. Direct content-href attribute on shreddit-post if it points to an image
    const contentHref = node.getAttribute ? node.getAttribute("content-href") : null;
    if (contentHref && (contentHref.endsWith(".png") || contentHref.endsWith(".jpg") || contentHref.endsWith(".jpeg") || contentHref.includes("redd.it"))) {
      images.push(contentHref);
    }

    if (images.length > 0) return [...new Set(images)].slice(0, 4);
    return super.extractImages(node);
  }
}

window.RedditAdapter = RedditAdapter;
