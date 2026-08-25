// Full-window image viewer (lightbox) for diagnostic figures. Plots used to
// open as raw PNGs in a new browser tab, which lost the caption, the step it
// came from and your place in the dashboard. This keeps them in the page:
// arrow keys / buttons walk a set of figures, click toggles fit-to-window vs
// actual size, Esc closes.
//
// Figures the runner could serialise (see fig_spec.py) also carry a plot spec,
// and open as an interactive plotly chart instead of the PNG — zoom, pan,
// hover, legend toggling. The toolbar switches back to the image, and figures
// with no spec only ever show the image.

const Viewer = {
  el: null,
  imgEl: null,
  items: [],   // [{fname, caption}]
  index: 0,

  // Build the overlay once, lazily, and keep it in the DOM afterwards.
  _ensure() {
    if (Viewer.el) return;
    const el = document.createElement('div');
    el.className = 'viewer hidden';
    el.innerHTML =
      `<div class="viewer-bar">` +
      `<span class="viewer-caption"></span>` +
      `<span class="viewer-note"></span>` +
      `<span class="viewer-count"></span>` +
      `<button class="viewer-toggle hidden"></button>` +
      `<a class="viewer-raw" target="_blank" rel="noopener" title="Open the PNG in a new tab">` +
      `${Icon.svg('external', 14)}</a>` +
      `<button class="viewer-close" title="Close (Esc)">${Icon.svg('close', 16)}</button>` +
      `</div>` +
      `<div class="viewer-stage">` +
      `<button class="viewer-nav prev" title="Previous (←)">${Icon.svg('right', 22)}</button>` +
      `<img class="viewer-img" alt="" />` +
      `<div class="viewer-plot hidden"></div>` +
      `<button class="viewer-nav next" title="Next (→)">${Icon.svg('right', 22)}</button>` +
      `</div>`;
    document.body.appendChild(el);
    Viewer.el = el;
    Viewer.imgEl = el.querySelector('.viewer-img');
    Viewer.plotEl = el.querySelector('.viewer-plot');

    el.querySelector('.viewer-close').onclick = Viewer.close;
    el.querySelector('.viewer-toggle').onclick = (e) => {
      e.stopPropagation();
      Viewer.interactive = !Viewer.interactive;
      Viewer.show();
    };
    // Dragging inside a chart must not reach the backdrop's close handler.
    Viewer.plotEl.onclick = (e) => e.stopPropagation();
    el.querySelector('.prev').onclick = (e) => { e.stopPropagation(); Viewer.step(-1); };
    el.querySelector('.next').onclick = (e) => { e.stopPropagation(); Viewer.step(1); };
    // Click the image to zoom; click the backdrop around it to close.
    Viewer.imgEl.onclick = (e) => { e.stopPropagation(); Viewer.toggleZoom(); };
    el.querySelector('.viewer-stage').onclick = Viewer.close;
    document.addEventListener('keydown', Viewer._onKey);
  },

  _onKey(e) {
    if (!Viewer.el || Viewer.el.classList.contains('hidden')) return;
    if (e.key === 'Escape') { Viewer.close(); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { Viewer.step(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { Viewer.step(1); e.preventDefault(); }
  },

  url(fname) { return '/api/run/figure/' + encodeURIComponent(fname); },

  // Captured filenames restart at fig_001.png every run, so a bare URL can be
  // served from cache as a *previous* run's image. Mint a unique URL once, when
  // the figure is captured — not per render, which would defeat caching
  // entirely and re-fetch on every repaint.
  _seq: 0,
  freshUrl(fname) { return Viewer.url(fname) + '?v=' + (++Viewer._seq); },

  // The URL to display a captured figure with: its minted one where present.
  src(fig) { return fig.url || Viewer.url(fig.fname); },

  // `items` is the set to page through; `index` the one to show first.
  open(items, index = 0) {
    if (!items || !items.length) return;
    Viewer._ensure();
    Viewer.items = items;
    Viewer.index = Math.max(0, Math.min(index, items.length - 1));
    Viewer.el.classList.remove('hidden');
    Viewer.show();
  },

  // Whether to open a spec-carrying figure interactively. Sticky across
  // figures, so paging through a set does not keep flipping mode.
  interactive: true,

  show() {
    const it = Viewer.items[Viewer.index];
    if (!it) return;
    Viewer.el.querySelector('.viewer-caption').textContent = it.caption || it.fname;
    Viewer.el.querySelector('.viewer-count').textContent =
      Viewer.items.length > 1 ? `${Viewer.index + 1} / ${Viewer.items.length}` : '';
    Viewer.el.querySelector('.viewer-raw').href = Viewer.src(it);
    const multi = Viewer.items.length > 1;
    Viewer.el.querySelectorAll('.viewer-nav').forEach((b) =>
      b.classList.toggle('hidden', !multi));

    const toggle = Viewer.el.querySelector('.viewer-toggle');
    toggle.classList.toggle('hidden', !it.spec);
    toggle.textContent = Viewer.interactive ? 'Image' : 'Interactive';
    toggle.title = Viewer.interactive
      ? 'Show the original matplotlib image'
      : 'Redraw this plot so it can be zoomed and panned';
    Viewer.note('');

    // Always tear the old chart down first: its WebGL contexts are a limited
    // resource, and paging through a set would otherwise pile them up.
    Plot.purge(Viewer.plotEl);
    if (it.spec && Viewer.interactive) {
      Viewer.showPlot(it);
    } else {
      Viewer.showImage(it);
    }
  },

  showImage(it) {
    Viewer.plotEl.classList.add('hidden');
    Viewer.imgEl.classList.remove('hidden', 'zoomed');
    Viewer.imgEl.src = Viewer.src(it);
    Viewer.imgEl.alt = it.caption || it.fname;
  },

  // Draw the interactive version. Anything that goes wrong — spec missing,
  // plotly unavailable — silently falls back to the image the run already has.
  showPlot(it) {
    const token = ++Viewer._token;
    Viewer.imgEl.classList.add('hidden');
    Viewer.plotEl.classList.remove('hidden');
    Viewer.plotEl.textContent = 'Loading plot…';
    Plot.fetchSpec(it.spec)
      .then((spec) => {
        if (token !== Viewer._token) return; // paged on while this was loading
        return Plot.render(Viewer.plotEl, spec, { name: it.spec }).then(() => {
          if (spec.thinned) {
            Viewer.note('thinned for display — the image is full resolution');
          }
        });
      })
      .catch(() => {
        if (token !== Viewer._token) return;
        Viewer.showImage(it);
        Viewer.note('interactive view unavailable');
      });
  },

  // Guards against a slow spec landing after the user has paged to another
  // figure, which would otherwise draw the wrong plot.
  _token: 0,

  note(text) {
    const el = Viewer.el && Viewer.el.querySelector('.viewer-note');
    if (el) el.textContent = text;
  },

  step(dir) {
    if (Viewer.items.length < 2) return;
    Viewer.index = (Viewer.index + dir + Viewer.items.length) % Viewer.items.length;
    Viewer.show();
  },

  toggleZoom() { Viewer.imgEl.classList.toggle('zoomed'); },

  close() {
    if (!Viewer.el) return;
    Viewer.el.classList.add('hidden');
    Viewer._token += 1;   // abandon any spec still loading
    Plot.purge(Viewer.plotEl);
  },

  // A clickable thumbnail/card for one figure that opens the viewer on `items`.
  // Shared by the Plots tab gallery and the paused-step review panel.
  card(items, index, { caption = true, cls = '' } = {}) {
    const it = items[index];
    const fig = document.createElement('figure');
    fig.className = 'plot-card' + (cls ? ' ' + cls : '');
    const img = document.createElement('img');
    img.src = Viewer.src(it);
    img.alt = it.caption || it.fname;
    img.loading = 'lazy';
    fig.appendChild(img);
    if (caption && it.caption) {
      const cap = document.createElement('figcaption');
      cap.textContent = it.caption;
      fig.appendChild(cap);
    }
    fig.onclick = () => Viewer.open(items, index);
    if (it.spec) fig.classList.add('has-plot');
    fig.title = it.spec
      ? 'Click to open — this plot can be zoomed and panned'
      : 'Click to view full size';
    return fig;
  },
};
