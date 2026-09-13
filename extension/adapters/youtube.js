/**
 * youtube.js
 * ----------
 * Platform adapter for YouTube (youtube.com).
 * Targets comments (ytd-comment-thread-renderer, ytd-comment-view-model)
 * and home/search feed cards (ytd-rich-item-renderer, ytd-video-renderer).
 */

class YouTubeAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("YouTube", ["youtube.com"]);
  }

  getPostSelector() {
    return "ytd-comment-thread-renderer, ytd-comment-view-model, ytd-rich-item-renderer, ytd-video-renderer, ytd-compact-video-renderer";
  }

  getComposeSelector() {
    return "div#contenteditable-root[contenteditable='true'], ytd-commentbox textarea, input#search, ytd-searchbox input, input[name='search_query']";
  }

  extractText(node) {
    // Comment text or video title
    const commentEl = node.querySelector("#content-text, yt-attributed-string#content-text");
    const titleEl = node.querySelector("#video-title, #video-title-link");
    if (commentEl) return (commentEl.innerText || "").trim().slice(0, 2500);
    if (titleEl) return (titleEl.innerText || "").trim().slice(0, 2500);
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    const thumbEls = node.querySelectorAll("yt-image img, #thumbnail img");
    thumbEls.forEach((img) => {
      if (img.src && img.src.startsWith("http") && !img.src.includes("avatar")) {
        images.push(img.src);
      }
    });
    if (images.length > 0) return images.slice(0, 4);
    return super.extractImages(node);
  }
}

window.YouTubeAdapter = YouTubeAdapter;
