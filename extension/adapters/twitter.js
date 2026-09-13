/**
 * twitter.js
 * ----------
 * Platform adapter for Twitter / X (twitter.com, x.com).
 * Targets tweets, replies, user handles, media containers, and tweet compose boxes.
 */

class TwitterAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("Twitter/X", ["twitter.com", "x.com"]);
  }

  getPostSelector() {
    return "article[data-testid='tweet'], div[data-testid='cellInnerDiv'] article";
  }

  getComposeSelector() {
    return "[data-testid='tweetTextarea_0'], [data-testid='tweetTextarea_0_label'], div[role='textbox'][data-testid*='tweet'], input[data-testid='SearchBox_Search_Input'], input[role='combobox'][placeholder*='Search' i]";
  }

  extractText(node) {
    const textNode = node.querySelector("[data-testid='tweetText']");
    if (textNode) return (textNode.innerText || "").trim().slice(0, 2500);
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const photoContainers = Array.from(node.querySelectorAll("[data-testid='tweetPhoto'] img"));
    if (photoContainers.length > 0) {
      return photoContainers
        .map((img) => img.src)
        .filter((src) => src && src.startsWith("http"));
    }
    return super.extractImages(node);
  }

  extractPostData(node) {
    const base = super.extractPostData(node);
    const authorEl = node.querySelector("[data-testid='User-Name']");
    const author = authorEl ? authorEl.innerText.split("\n")[0] : null;
    return { ...base, author };
  }
}

window.TwitterAdapter = TwitterAdapter;
