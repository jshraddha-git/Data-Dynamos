/**
 * generic.js
 * ----------
 * Generic fallback adapter for blogs, forums, and unsupported sites.
 * Uses broad semantic HTML5 elements and accessibility role attributes.
 */

class GenericAdapter extends window.BasePlatformAdapter {
  constructor() {
    super("Generic", []);
  }

  matches(_hostname) {
    return true; // Fallback matches any host
  }

  getPostSelector() {
    return "article, [role='article'], .post, .comment, .feed-item, .card";
  }

  getComposeSelector() {
    return "textarea, [contenteditable='true'], input[type='text'], div[role='textbox']";
  }
}

window.GenericAdapter = GenericAdapter;
