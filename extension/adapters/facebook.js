/**
 * facebook.js
 * -----------
 * Platform adapter for Facebook (facebook.com).
 * Targets feed articles, post units, and comments.
 */

class FacebookAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("Facebook", ["facebook.com"]);
  }

  getPostSelector() {
    return "div[role='article'], div[data-pagelet*='FeedUnit'], div.x1yztbdb";
  }

  getComposeSelector() {
    return "div[role='textbox'][contenteditable='true'], div[aria-label*='Write a comment']";
  }

  extractText(node) {
    const textEls = node.querySelectorAll("div[dir='auto'][style*='text-align: start'], div[data-ad-preview='message']");
    if (textEls.length > 0) {
      const combined = Array.from(textEls).map((el) => el.innerText.trim()).join("\n");
      if (combined) return combined.slice(0, 2500);
    }
    return (node.innerText || "").trim().slice(0, 2500);
  }

  extractImages(node) {
    const images = [];
    const mediaEls = node.querySelectorAll("img.x1ey2m1c, img[src*='fbcdn']");
    mediaEls.forEach((img) => {
      if (img.src && img.src.startsWith("http") && !img.src.includes("rsrc.php")) {
        images.push(img.src);
      }
    });
    if (images.length > 0) return images.slice(0, 4);
    return super.extractImages(node);
  }
}

window.FacebookAdapter = FacebookAdapter;
