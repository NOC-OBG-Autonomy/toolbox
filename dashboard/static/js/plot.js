// Redraw a captured matplotlib figure interactively with plotly.
//
// The runner writes a plot spec (dashboard/fig_spec.py) beside each diagnostic
// PNG, holding the figure's actual x/y arrays. This turns one into stacked
// plotly panels — one per matplotlib axes — so the plot can be box-zoomed,
// panned and hovered, and series toggled from the legend. Panels the step drew
// with sharex=True have their x-ranges linked, as they were in matplotlib.
//
// Only figures fig_spec could represent faithfully get a spec at all; the
// viewer falls back to the PNG for the rest, so nothing here has to guess.

const Plot = {
  // plotly is ~4.5 MB, and most sessions never open an interactive plot — load
  // it on first use rather than on every page load. Concurrent callers share
  // the one in-flight promise.
  _loading: null,
  load() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (Plot._loading) return Plot._loading;
    Plot._loading = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/vendor/plotly.min.js';
      s.onload = () => resolve(window.Plotly);
      s.onerror = () => { Plot._loading = null; reject(new Error('plotly failed to load')); };
      document.head.appendChild(s);
    });
    return Plot._loading;
  },

  specUrl(name) { return '/api/run/figspec/' + encodeURIComponent(name); },

  dataUrl(name, panel, trace, xMin, xMax) {
    return '/api/run/figdata/' + encodeURIComponent(name)
      + '?panel=' + panel + '&trace=' + trace
      + '&x_min=' + xMin + '&x_max=' + xMax;
  },

  // Fetched specs are cached by filename: figure names are unique within a run
  // and the whole directory is cleared when the next run starts.
  _cache: {},
  fetchSpec(name) {
    if (Plot._cache[name]) return Plot._cache[name];
    Plot._cache[name] = fetch(Plot.specUrl(name)).then((r) => {
      if (!r.ok) throw new Error('spec ' + r.status);
      return r.json();
    }).catch((err) => { delete Plot._cache[name]; throw err; });
    return Plot._cache[name];
  },

  // Above this many points a trace is drawn with WebGL (scattergl). SVG is
  // sharper and prints better, so keep it for the small ones.
  GL_THRESHOLD: 4000,

  // Debounce a box-zoom/pan before asking for full-resolution data: zooming
  // is a drag gesture that fires many relayout events, and each one aborts
  // the fetch before it, so only the settled range is ever actually fetched.
  LOD_DEBOUNCE_MS: 300,

  // The figdata endpoint's wire format: uint32 header length, JSON header,
  // then each array's raw little-endian bytes. buffer.slice() copies into a
  // fresh (aligned) ArrayBuffer first, so the typed views are valid no matter
  // where the header length lands them -- a memcpy, not a parse, so still far
  // cheaper than JSON-parsing the same points would be.
  _parseFigData(buffer) {
    const headerLen = new DataView(buffer).getUint32(0, true);
    const header = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buffer, 4, headerLen)));
    let off = 4 + headerLen;
    const n = header.n;
    const x = new Float64Array(buffer.slice(off, off + n * 8)); off += n * 8;
    const y = new Float64Array(buffer.slice(off, off + n * 8)); off += n * 8;
    const color = header.has_color ? new Uint8Array(buffer.slice(off, off + n * 4)) : null;
    return { n, complete: header.complete, x, y, color };
  },

  // Epoch-ms floats -> ISO strings, the same date representation
  // dashboard/fig_spec.py's _values() emits for the initial spec, so a
  // restyled trace reads on a date axis exactly like the original one did.
  _isoDates(ms) {
    const out = new Array(ms.length);
    for (let i = 0; i < ms.length; i++) out[i] = isFinite(ms[i]) ? new Date(ms[i]).toISOString() : null;
    return out;
  },

  // Nx4 uint8 RGBA -> 'rgba(r,g,b,a)' strings, for a per-point marker.color.
  _rgbaStrings(rgba) {
    const out = new Array(rgba.length / 4);
    for (let i = 0; i < out.length; i++) {
      const o = i * 4;
      out[i] = 'rgba(' + rgba[o] + ',' + rgba[o + 1] + ',' + rgba[o + 2] + ',' + (rgba[o + 3] / 255) + ')';
    }
    return out;
  },

  trace(t) {
    const gl = t.x.length > Plot.GL_THRESHOLD;
    // A dense WebGL point cloud reads better a touch larger, same as
    // glider_playground's own scattergl traces (its defaultMarkerSize is 8).
    const marker = { size: Math.max(gl ? 5 : 3, t.size || 6), opacity: t.opacity };
    if (t.color) marker.color = t.color;
    return {
      type: gl ? 'scattergl' : 'scatter',
      mode: t.mode || 'lines',
      x: t.x,
      y: t.y,
      // A matplotlib label starting with "_" is one it hides from the legend.
      name: (t.label && !t.label.startsWith('_')) ? t.label : '',
      showlegend: !!(t.label && !t.label.startsWith('_')),
      line: { color: Array.isArray(t.color) ? undefined : t.color, width: t.width, dash: t.dash },
      marker,
      opacity: t.opacity,
      hovertemplate: '%{x}, %{y}<extra></extra>',
    };
  },

  // Render `spec` into `host`, one plotly div per panel, stacked vertically.
  // `name` is the figure filename (e.g. "fig_001.json"), needed to fetch
  // full-resolution data for any trace the spec marked "lod": true; omit it
  // (or leave a figure PNG-only) and zooming just re-scales the thinned spec,
  // as before. Returns a promise resolving once every panel is drawn.
  render(host, spec, { theme = 'light', name = null } = {}) {
    return Plot.load().then((Plotly) => {
      host.innerHTML = '';
      const dark = theme === 'dark';
      const fg = dark ? '#d6dae0' : '#222';
      const grid = dark ? '#3a4048' : '#e2e5e9';

      // Share the stage between panels, the way the matplotlib subplots did.
      // Plotly sizes itself from the container at newPlot time, so the height
      // has to be a real number of pixels rather than a percentage.
      const avail = host.clientHeight || 600;

      // Reproduce the figure's own panel layout when every panel carries its
      // grid cell (unequal row heights, multi-column spans): absolutely place
      // each panel at its cell within a stage of the available height. Falls
      // back to an even vertical stack when any cell is missing.
      const gridLayout = spec.panels.every((p) => Array.isArray(p.cell));
      let stage = host;
      if (gridLayout) {
        stage = document.createElement('div');
        stage.className = 'plot-grid';
        stage.style.cssText = 'position:relative;width:100%;height:' + avail + 'px;';
        host.appendChild(stage);
      }
      const panelHeight = Math.max(180, Math.floor(avail / spec.panels.length));

      const divs = [];
      spec.panels.forEach((panel, i) => {
        const div = document.createElement('div');
        div.className = 'plotly-panel';
        if (gridLayout) {
          const [l, t, w, h] = panel.cell;
          div.style.cssText = 'position:absolute;left:' + (l * 100) + '%;top:'
            + (t * 100) + '%;width:' + (w * 100) + '%;height:' + (h * 100) + '%;';
        } else {
          div.style.height = panelHeight + 'px';
        }
        stage.appendChild(div);
        divs.push(div);

        const axis = (label, range, scale, extra) => Object.assign({
          title: { text: label || '', font: { size: 12, color: fg } },
          range: range && range.every((v) => v !== null) ? range : undefined,
          type: scale === 'log' ? 'log' : undefined,
          gridcolor: grid, zerolinecolor: grid,
          linecolor: grid, tickfont: { size: 11, color: fg },
          automargin: true,
        }, extra || {});

        const layout = {
          // Only the top panel carries the figure title; a per-axes title on
          // every panel is how matplotlib does it, and plotly has room for it.
          title: (i === 0 && (spec.suptitle || panel.title))
            ? { text: spec.suptitle || panel.title, font: { size: 14, color: fg } }
            : (panel.title ? { text: panel.title, font: { size: 13, color: fg } } : undefined),
          xaxis: axis(panel.xlabel, panel.xlim, panel.xscale,
            panel.xdate ? { type: 'date' } : null),
          // A depth axis arrives as [deep, shallow]; plotly reads a descending
          // range as an inverted axis, so it needs no special casing. TIME can
          // sit on y too (an index-vs-TIME QC plot), hence the date type here.
          yaxis: axis(panel.ylabel, panel.ylim, panel.yscale,
            panel.ydate ? { type: 'date' } : null),
          showlegend: panel.legend,
          legend: {
            title: { text: panel.legend_title || '', font: { size: 11, color: fg } },
            font: { size: 11, color: fg }, bgcolor: 'rgba(0,0,0,0)',
          },
          margin: { l: 60, r: 20, t: panel.title || i === 0 ? 40 : 12, b: 45 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: fg },
          hovermode: 'closest',
          dragmode: 'zoom',
          // axhline/axvline reference lines (range bounds, min/max, correction
          // levels): full-span shapes pinned to one data value, so they track
          // zoom/pan instead of being points that scroll away.
          shapes: (panel.reflines || []).map((r) => ({
            type: 'line',
            xref: r.axis === 'x' ? 'x' : 'paper',
            yref: r.axis === 'y' ? 'y' : 'paper',
            x0: r.axis === 'x' ? r.value : 0, x1: r.axis === 'x' ? r.value : 1,
            y0: r.axis === 'y' ? r.value : 0, y1: r.axis === 'y' ? r.value : 1,
            line: { color: r.color, width: r.width || 1, dash: r.dash || 'dash' },
            opacity: r.opacity == null ? 1 : r.opacity,
            layer: 'below',
          })),
        };

        Plotly.newPlot(div, panel.traces.map(Plot.trace), layout, {
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          modeBarButtonsToRemove: ['select2d', 'lasso2d'],
          toImageButtonOptions: { format: 'png', scale: 2 },
        });
      });

      Plot._linkX(divs, spec.panels);
      Plot._trackResize(host);
      if (name) Plot._attachLod(divs, spec, name);
      return divs;
    });
  },

  // Wire up zoomed-in full-resolution fetching for panels with an "lod"
  // trace. Each div tracks, per trace index, the range it last fetched an
  // exact ("complete") answer for -- zooming further inside that range needs
  // no refetch, matching the range the server already gave everything it had.
  _attachLod(divs, spec, name) {
    divs.forEach((div, panelIdx) => {
      const panel = spec.panels[panelIdx];
      const lodTraces = panel.traces
        .map((t, i) => (t.lod ? i : -1))
        .filter((i) => i >= 0);
      if (!lodTraces.length) return;

      const original = lodTraces.map((i) => panel.traces[i]);
      const covered = {}; // traceIdx -> {lo, hi} of the last complete fetch
      let timer = null;
      let controller = null;
      let live = true; // false once purge() tears this div down

      // A restyle/update targeting a div Plotly has since purged throws inside
      // Plotly's internals; swallow it (there is nothing left to update).
      const safeRestyle = (attrs, idx) => {
        if (!live) return;
        try { window.Plotly.restyle(div, attrs, idx); } catch (_err) { /* div gone */ }
      };
      const safeUpdate = (dataAttrs, layoutAttrs, idx) => {
        if (!live) return;
        try { window.Plotly.update(div, dataAttrs, layoutAttrs, idx); } catch (_err) { /* div gone */ }
      };

      // Plotly reports a date axis's relayout range as a naive-UTC string
      // ("YYYY-MM-DD HH:MM:SS.sss", no zone marker) -- the same convention
      // _values() used to write the original trace. `new Date()` on a
      // space-separated string with no zone is parsed as *local* time by most
      // browsers, which would silently offset every fetch by the browser's
      // UTC offset (and look like scrambled data). Force UTC explicitly.
      const toMs = (v) => {
        if (!panel.xdate) return Number(v);
        const s = String(v);
        const hasZone = /[zZ]|[+-]\d\d:\d\d$/.test(s);
        return new Date(hasZone ? s : s.replace(' ', 'T') + 'Z').getTime();
      };
      // The panel's full original extent, in the same units as toMs() -- any
      // requested range at least this wide is "zoomed all the way out".
      const origLo = panel.xlim && panel.xlim[0] != null ? toMs(panel.xlim[0]) : -Infinity;
      const origHi = panel.xlim && panel.xlim[1] != null ? toMs(panel.xlim[1]) : Infinity;

      // Restore data and pin the axis back to the panel's real extent in one
      // atomic Plotly.update call -- not a restyle followed by a separate
      // relayout. Split in two, a sibling panel synced by _linkX's own
      // programmatic relayout could have its axis autorange computed from the
      // trace's *still-zoomed* data (restyle not applied yet), leaving that
      // panel's range stuck narrow even after its data caught up -- which is
      // what "reset doesn't do all, just one" looked like. Setting an explicit
      // range instead of autorange sidesteps needing that computation at all.
      const restoreOriginal = () => {
        if (controller) controller.abort();
        lodTraces.forEach((traceIdx) => { delete covered[traceIdx]; });
        safeUpdate(
          { x: original.map((t) => t.x), y: original.map((t) => t.y),
            'marker.color': original.map((t) => t.color) },
          { 'xaxis.range': panel.xlim, 'xaxis.autorange': false },
          lodTraces,
        );
      };

      div.on('plotly_relayout', (ev) => {
        clearTimeout(timer);
        // Drag/scroll zoom emits indexed keys; the modebar zoom-out button and
        // programmatic relayouts (e.g. _linkX syncing a sibling panel) emit a
        // single 'xaxis.range' array instead -- both mean the range changed.
        const range = ('xaxis.range[0]' in ev) ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']]
          : (Array.isArray(ev['xaxis.range']) ? ev['xaxis.range'] : null);
        const reset = ev['xaxis.autorange'] === true;
        if (!range && !reset) return; // some other relayout (e.g. a resize)

        const xMin = reset ? origLo : toMs(range[0]);
        const xMax = reset ? origHi : toMs(range[1]);

        // Reset button/double-click, or zooming out at least to the original
        // extent either way: show the original data directly rather than
        // fetching (which would need no thinning applied beyond the original
        // anyway, and avoids depending on exactly how "reset" was triggered).
        if (reset || (xMin <= origLo && xMax >= origHi)) {
          restoreOriginal();
          return;
        }

        timer = setTimeout(() => {
          if (!live) return;
          if (controller) controller.abort();
          controller = new AbortController();
          const { signal } = controller;

          lodTraces.forEach((traceIdx) => {
            const have = covered[traceIdx];
            if (have && xMin >= have.lo && xMax <= have.hi) return; // already exact here

            fetch(Plot.dataUrl(name, panelIdx, traceIdx, xMin, xMax), { signal })
              .then((r) => { if (!r.ok) throw new Error('figdata ' + r.status); return r.arrayBuffer(); })
              .then((buf) => {
                if (!live) return;
                const d = Plot._parseFigData(buf);
                const x = panel.xdate ? Plot._isoDates(d.x) : d.x;
                const y = panel.ydate ? Plot._isoDates(d.y) : d.y;
                const restyle = { x: [x], y: [y] };
                if (d.color) restyle['marker.color'] = [Plot._rgbaStrings(d.color)];
                safeRestyle(restyle, [traceIdx]);
                if (d.complete) covered[traceIdx] = { lo: xMin, hi: xMax };
                else delete covered[traceIdx];
              })
              .catch((err) => { if (err.name !== 'AbortError') { /* stay on current data */ } });
          });
        }, Plot.LOD_DEBOUNCE_MS);
      });

      // Let Plot.purge() stop any pending debounce/fetch and block further
      // restyles once this div is torn down, instead of leaving them to fire
      // later against a purged Plotly instance.
      div._lodCleanup = () => {
        live = false;
        clearTimeout(timer);
        if (controller) controller.abort();
      };
    });
  },

  // Re-fit the panels when the window changes size. One observer per host,
  // reused across figures; Plot.purge empties the host, not this.
  _trackResize(host) {
    if (host._plotObserver || typeof ResizeObserver === 'undefined') return;
    let pending = null;
    host._plotObserver = new ResizeObserver(() => {
      clearTimeout(pending);
      pending = setTimeout(() => {
        const panels = host.querySelectorAll('.plotly-panel');
        if (!panels.length || !window.Plotly) return;
        const grid = host.querySelector('.plot-grid');
        if (grid) {
          // Panels are %-positioned within the stage; resize the stage and let
          // them follow, then re-fit each plotly plot to its new box.
          grid.style.height = (host.clientHeight || 600) + 'px';
        } else {
          const h = Math.max(180, Math.floor((host.clientHeight || 600) / panels.length));
          panels.forEach((d) => { d.style.height = h + 'px'; });
        }
        panels.forEach((d) => window.Plotly.Plots.resize(d));
      }, 120);
    });
    host._plotObserver.observe(host);
  },

  // Keep the x-ranges of panels in the same sharex group in step. The guard
  // stops the relayout we trigger from bouncing back and looping.
  _linkX(divs, panels) {
    let syncing = false;
    divs.forEach((div, i) => {
      div.on('plotly_relayout', (ev) => {
        if (syncing) return;
        const range = ('xaxis.range[0]' in ev)
          ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']]
          : (ev['xaxis.autorange'] ? null : undefined);
        if (range === undefined) return;
        syncing = true;
        divs.forEach((other, j) => {
          if (j === i || panels[j].share_x !== panels[i].share_x) return;
          window.Plotly.relayout(other, range
            ? { 'xaxis.range': range }
            : { 'xaxis.autorange': true });
        });
        syncing = false;
      });
    });
  },

  // Free the WebGL contexts a rendered figure holds. Browsers cap how many are
  // live at once, so leaving them behind eventually blanks new plots.
  purge(host) {
    if (!window.Plotly || !host) return;
    host.querySelectorAll('.plotly-panel').forEach((d) => {
      // Stop any pending LOD debounce/fetch before Plotly.purge tears the div
      // down -- a restyle landing afterwards would hit a dead Plotly instance.
      if (d._lodCleanup) d._lodCleanup();
      // Grab each live context before Plotly detaches its canvas: Plotly.purge
      // tears the plot down but the browser keeps the WebGL context alive until
      // GC, and with a context per panel and a ~16 cap, paging through figures
      // eventually blanks new plots. Drop them explicitly instead of waiting.
      const lost = [];
      d.querySelectorAll('canvas').forEach((c) => {
        const gl = c.getContext('webgl2') || c.getContext('webgl');
        if (gl) lost.push(gl.getExtension('WEBGL_lose_context'));
      });
      window.Plotly.purge(d);
      lost.forEach((ext) => ext && ext.loseContext());
    });
    host.innerHTML = '';
  },
};
