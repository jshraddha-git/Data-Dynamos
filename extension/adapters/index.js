/**
 * index.js (Adapter Registry)
 * ----------------------------
 * Selects and initializes the active platform adapter based on the current window.location.hostname.
 */

(() => {
  const adapters = [
    new window.TwitterAdapter(),
    new window.RedditAdapter(),
    new window.InstagramAdapter(),
    new window.FacebookAdapter(),
    new window.TikTokAdapter(),
    new window.LinkedInAdapter(),
    new window.YouTubeAdapter(),
    new window.GenericAdapter(), // Fallback always last
  ];

  function getActiveAdapter(hostname = window.location.hostname) {
    const active = adapters.find((a) => a.matches(hostname));
    return active || adapters[adapters.length - 1];
  }

  window.WellbeingAdapterRegistry = {
    getActiveAdapter,
    adapters,
  };
})();
