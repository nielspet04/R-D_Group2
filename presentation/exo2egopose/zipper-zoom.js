/* Slide 14: enlarge the same zipper frame for a direct pose comparison. */
(() => {
  const panel = document.querySelector('#zipper-model-comparison');
  if (!panel) return;
  const slide = panel.closest('section');
  let returnFocus;
  let zoomShown = false;
  const printing = new URLSearchParams(location.search).has('print-pdf');

  function canStep() {
    return !printing && Reveal.getCurrentSlide() === slide &&
      !Reveal.isOverview() && !Reveal.isPaused() &&
      !document.body.classList.contains('editing') &&
      !document.body.classList.contains('panel-open');
  }

  function replay() {
    panel.classList.remove('is-zoomed');
    void panel.offsetWidth;
    panel.classList.add('is-zoomed');
  }
  function close(restoreFocus = true) {
    if (panel.hidden) return;
    panel.hidden = true;
    panel.classList.remove('is-zoomed');
    if (restoreFocus && returnFocus?.isConnected) returnFocus.focus();
  }
  function show() {
    returnFocus = document.activeElement;
    panel.hidden = false;
    panel.focus({ preventScroll: true });
    replay();
  }
  function consumeForwardStep() {
    if (!canStep()) return false;
    if (!zoomShown) {
      zoomShown = true;
      show();
      return true;
    }
    close(false);
    return false;
  }

  // Capture Reveal's controls before their own handlers advance the slide.
  for (const name of ['click', 'touchstart']) {
    document.addEventListener(name, event => {
      const next = event.target.closest?.('.navigate-right, .navigate-next, .navigate-down');
      if (next && consumeForwardStep()) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    }, { capture: true, passive: false });
  }

  // Speaker-view and programmatic navigation use the same extra step.
  function installNavigation() {
    for (const name of ['next', 'right', 'down', 'navigateNext', 'navigateRight', 'navigateDown']) {
      const original = Reveal[name];
      if (typeof original !== 'function') continue;
      Reveal[name] = function (...args) {
        if (!consumeForwardStep()) return original.apply(this, args);
      };
    }
  }
  if (Reveal.isReady()) installNavigation();
  else Reveal.on('ready', installNavigation);
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.altKey || event.metaKey || event.shiftKey ||
        event.target.closest?.('input, textarea, [contenteditable="true"]')) return;
    if (canStep() && ['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'n', 'N', 'j', 'J', 'l', 'L'].includes(event.key)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!consumeForwardStep()) Reveal.next();
      return;
    }
    if (panel.hidden) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
    } else if (event.key === 'Tab') {
      event.preventDefault();
      panel.focus({ preventScroll: true });
    } else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'p', 'P', 'h', 'H', 'k', 'K'].includes(event.key)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
    }
  }, true);
  Reveal.on('slidechanged', () => { close(false); zoomShown = false; });
  Reveal.on('overviewshown', () => close(false));
  panel.addEventListener('click', event => event.stopPropagation());
})();
