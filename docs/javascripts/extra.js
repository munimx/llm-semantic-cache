/**
 * extra.js — Recallm docs: custom copy buttons
 *
 * Replaces MkDocs Material's default clipboard buttons with custom styled
 * terminal-style buttons that match the site's design language.
 * Hero code panel gets no copy button (decorative code block).
 */

function initCustomCopyButtons() {
  // Remove copy button from the hero code panel — it's decorative
  document.querySelectorAll('.hero-code-panel .md-clipboard').forEach(function(btn) {
    btn.remove();
  });

  // Replace all remaining default .md-clipboard buttons with custom versions
  document.querySelectorAll('.md-clipboard').forEach(function(defaultBtn) {
    var highlight = defaultBtn.closest('.highlight');
    if (!highlight) return;

    var pre = highlight.querySelector('pre');
    if (!pre) return;

    var btn = document.createElement('button');
    btn.className = 'recallm-copy';
    btn.title = 'Copy to clipboard';
    btn.setAttribute('aria-label', 'Copy to clipboard');
    btn.textContent = '⧉';

    btn.addEventListener('click', function() {
      var text = (pre.innerText || pre.textContent || '').trim();
      navigator.clipboard.writeText(text).then(function() {
        btn.textContent = '✓';
        btn.classList.add('recallm-copy--done');
        setTimeout(function() {
          btn.textContent = '⧉';
          btn.classList.remove('recallm-copy--done');
        }, 1500);
      }).catch(function() {
        // Fallback for browsers without clipboard API
        var range = document.createRange();
        range.selectNodeContents(pre);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand('copy');
        sel.removeAllRanges();
        btn.textContent = '✓';
        setTimeout(function() { btn.textContent = '⧉'; }, 1500);
      });
    });

    defaultBtn.replaceWith(btn);
  });
}

// MkDocs Material SPA navigation hook (fires after every page load/nav)
if (typeof document$ !== 'undefined') {
  document$.subscribe(function() {
    // Clipboard buttons are injected by MkDocs Material JS — give it a tick
    setTimeout(initCustomCopyButtons, 50);
  });
} else {
  // Fallback for initial load without document$ available yet
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      setTimeout(initCustomCopyButtons, 300);
    });
  } else {
    setTimeout(initCustomCopyButtons, 300);
  }
}
