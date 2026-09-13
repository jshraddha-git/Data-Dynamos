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
    return "shreddit-post, div[data-testid='post-container'], shreddit-comment, .Comment, .entry, div[data-testid*='search-post'], div[data-testid*='search-result'], div[data-testid='search-slink-post'], search-telemetry-tracker, [data-testid='search-unit'], [data-testid='search-post-unit'], .search-result, .search-result-link, community-post";
  }

  getComposeSelector() {
    return "shreddit-composer, div[role='textbox'][contenteditable='true'], textarea[name='text'], input[type='search'], input[name='q'], [role='searchbox'], faceplate-search-input input, reddit-search-large input, reddit-header-search-bar input, input[placeholder*='Search' i], input[aria-label*='Search' i], input[data-testid*='search']";
  }

  extractText(node) {
    let title = (node.getAttribute && node.getAttribute("post-title")) || "";
    if (!title) {
      const titleEl = node.querySelector("[slot='title'], h1, h2, h3, a[data-testid='post-title'], a[slot='full-post-link'], p[data-testid='post-title'], a[data-testid*='search-post-link'], [data-testid='post-title-text']");
      if (titleEl) title = (titleEl.innerText || titleEl.textContent || "").trim();
    }

    const bodyEl = node.querySelector("[slot='text-body'], [slot='comment'], div[data-testid='post-content'], .usertext-body, div[id$='-post-rtjson-content'], [data-click-id='text'], .md, p, [data-testid*='search-post-snippet'], [data-testid*='post-snippet']");
    const body = bodyEl ? (bodyEl.innerText || bodyEl.textContent || "").trim() : "";

    let shadowText = "";
    if (node.shadowRoot) {
      shadowText = (node.shadowRoot.innerText || node.shadowRoot.textContent || "").trim();
    }

    const full = `${title} ${body} ${shadowText} ${node.innerText || node.textContent || ""}`.replace(/\s+/g, " ").trim();
    return full.slice(0, 2500);
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
