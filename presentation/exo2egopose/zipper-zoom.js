/* Slide 14: enlarge the same zipper frame for a direct pose comparison. */
(() => {
  const trigger = document.querySelector('[data-zipper-zoom-open]');
  const panel = document.querySelector('#zipper-model-comparison');
  if (!trigger || !panel) return;
  const closeButton = panel.querySelector('[data-zipper-zoom-close]');
  const replayButton = panel.querySelector('[data-zipper-zoom-replay]');
  let returnFocus;

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
  trigger.addEventListener('click', event => {
    if (document.body.classList.contains('editing')) return;
    event.stopPropagation();
    returnFocus = document.activeElement;
    panel.hidden = false;
    closeButton.focus({ preventScroll: true });
    replay();
  });
  closeButton.addEventListener('click', () => close());
  replayButton.addEventListener('click', replay);
  document.addEventListener('keydown', event => {
    if (panel.hidden) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
    } else if (event.key === 'Tab') {
      event.preventDefault();
      (document.activeElement === closeButton ? replayButton : closeButton).focus();
    } else if (event.key.startsWith('Arrow') || event.key === ' ') {
      // Leave slide navigation available; close when the slide changes.
      close(false);
    }
  }, true);
  Reveal.on('slidechanged', () => close(false));
  Reveal.on('overviewshown', () => close(false));
  panel.addEventListener('click', event => event.stopPropagation());
  // Keep the interactive overlay out of the static PDF and overview.
  if (new URLSearchParams(location.search).has('print-pdf')) trigger.hidden = true;
})();
