// Live RAM meter for a run. Each __PELAGOS_MEM__ marker (one per executed step)
// adds a point; the sparkline plots RSS across steps so transient per-step
// spikes — and their release — are visible at a glance. Per-step detail lives
// in each dot's hover tooltip.

const Mem = {
  points: [],  // {rss, stepPeak, added, data, label} per step, in run order
  peak: 0,     // running max RSS this run — kept only to scale the plot's y-axis

  // Shown immediately when a run starts (not hidden) so the meter feels live
  // the instant Run is pressed, rather than popping in once the first step's
  // __PELAGOS_MEM__ marker arrives.
  reset() {
    Mem.points = [];
    Mem.peak = 0;
    const meter = document.getElementById('mem-meter');
    if (meter) meter.classList.remove('hidden');
    const spark = document.getElementById('mem-spark');
    if (spark) spark.innerHTML = '';
    ['mem-cur', 'mem-peak', 'mem-data'].forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.textContent = '–';
    });
  },

  // "<rss>\t<runPeak>\t<data>\t<label>\t<stepPeak>\t<peakLabel>\t<stepStart>".
  // runPeak/peakLabel are consumed only for the plot scale / tooltip; trailing
  // fields are absent on older runners, so fall back gracefully. Values are MB.
  add(payload) {
    const parts = payload.split('\t');
    const rss = parseFloat(parts[0]);
    if (!isFinite(rss)) return;
    const runPeak = parseFloat(parts[1]);
    const data = parseFloat(parts[2]); // NaN when the field is empty
    const label = (parts[3] || '').trim();
    const stepPeak = isFinite(parseFloat(parts[4])) ? parseFloat(parts[4]) : rss;
    const stepStart = parseFloat(parts[6]); // NaN on older runners
    // A step's own growth: how much RSS it added on top of what it inherited.
    // Without step-start, fall back to growth over the previous settle.
    const prev = Mem.points.length ? Mem.points[Mem.points.length - 1].rss : stepPeak;
    const added = Math.max(0, stepPeak - (isFinite(stepStart) ? stepStart : prev));
    Mem.peak = isFinite(runPeak) ? runPeak : Math.max(Mem.peak, rss);
    Mem.points.push({ rss, stepPeak, added, data: isFinite(data) ? data : null, label });
    document.getElementById('mem-meter').classList.remove('hidden');
    Mem.setNum('mem-cur', rss);
    Mem.setNum('mem-peak', Mem.peak);
    Mem.setNum('mem-data', data);
    Mem.render();
  },

  // MB -> "1.9 GB" / "512 MB". null -> "–".
  fmt(mb) {
    if (mb == null || !isFinite(mb)) return '–';
    return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
  },

  setNum(id, mb) {
    const el = document.getElementById(id);
    if (el) el.textContent = Mem.fmt(mb);
  },

  // Redraw the sparkline. The line is each step's *in-step* peak RSS (the true
  // high-water while it ran, which the boundary reading alone would miss), so a
  // step that briefly spikes shows up. The y-axis runs 0..run-peak so a drop
  // after a spike is obvious. Hover detail is a custom tooltip (see below),
  // not native <title>s: the dots are only ~2px, far too small a target to
  // reliably hover, so layout() instead tracks each point's x position and a
  // mousemove listener picks the nearest one regardless of exact cursor y.
  //
  // The viewBox is set to the SVG's actual rendered pixel size (not a fixed
  // 100-unit box) so 1 user unit == 1px in both axes. Otherwise
  // preserveAspectRatio="none" stretches x and y by different factors and
  // circles render as squashed ellipses.
  render() {
    const svg = document.getElementById('mem-spark');
    if (!svg) return;
    const H = 34, pad = 3;
    const W = Math.max(svg.clientWidth || 1, 60);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const n = Mem.points.length;
    const top = Mem.peak || 1;
    const x = (i) => (n <= 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1));
    const y = (mb) => H - pad - (mb / top) * (H - 2 * pad);
    const svgns = 'http://www.w3.org/2000/svg';
    svg.innerHTML = '';
    Mem._layout = { xs: Mem.points.map((_, i) => x(i)) };

    // Peak guide line, so the ceiling of the run is always marked.
    const guide = document.createElementNS(svgns, 'line');
    guide.setAttribute('x1', 0); guide.setAttribute('x2', W);
    guide.setAttribute('y1', y(top)); guide.setAttribute('y2', y(top));
    guide.setAttribute('class', 'mem-spark-peak');
    svg.appendChild(guide);

    if (n > 1) {
      const linePts = Mem.points.map((p, i) => `${x(i)},${y(p.stepPeak)}`).join(' ');
      const area = document.createElementNS(svgns, 'polygon');
      area.setAttribute('points', `${x(0)},${H - pad} ${linePts} ${x(n - 1)},${H - pad}`);
      area.setAttribute('class', 'mem-spark-area');
      svg.appendChild(area);

      const line = document.createElementNS(svgns, 'polyline');
      line.setAttribute('points', linePts);
      line.setAttribute('class', 'mem-spark-line');
      svg.appendChild(line);
    }

    Mem.points.forEach((p, i) => {
      const isPeak = p.stepPeak >= Mem.peak;
      const dot = document.createElementNS(svgns, 'circle');
      dot.setAttribute('cx', x(i));
      dot.setAttribute('cy', y(p.stepPeak));
      dot.setAttribute('r', isPeak ? 2.6 : 1.8);
      dot.setAttribute('class', isPeak ? 'mem-spark-dot peak' : 'mem-spark-dot');
      dot.dataset.i = i;
      svg.appendChild(dot);
    });
  },

  // Nearest point to a mouse event's x position, in SVG user units.
  nearestPoint(svg, clientX) {
    const xs = Mem._layout && Mem._layout.xs;
    if (!xs || !xs.length) return -1;
    const rect = svg.getBoundingClientRect();
    if (!rect.width) return -1;
    const viewBox = svg.viewBox.baseVal;
    const mx = (clientX - rect.left) * (viewBox.width / rect.width);
    let best = 0, bestDist = Infinity;
    xs.forEach((xi, i) => {
      const d = Math.abs(xi - mx);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  },

  showTooltip(i, clientX, clientY) {
    const svg = document.getElementById('mem-spark');
    const tip = document.getElementById('mem-tooltip');
    if (!svg || !tip) return;
    const p = Mem.points[i];
    if (!p) return;
    svg.querySelectorAll('.mem-spark-dot.active').forEach((d) => d.classList.remove('active'));
    const dot = svg.querySelector(`.mem-spark-dot[data-i="${i}"]`);
    if (dot) dot.classList.add('active');
    tip.innerHTML = `<b>${p.label || 'step ' + (i + 1)}</b>` +
      `<br>peak ${Mem.fmt(p.stepPeak)} &nbsp;+${Mem.fmt(p.added)} this step` +
      `<br>settled ${Mem.fmt(p.rss)}` +
      (p.data != null ? `<br>data ${Mem.fmt(p.data)}` : '');
    tip.style.left = `${clientX}px`;
    tip.style.top = `${clientY}px`;
    tip.classList.remove('hidden');
  },

  hideTooltip() {
    const tip = document.getElementById('mem-tooltip');
    if (tip) tip.classList.add('hidden');
    const svg = document.getElementById('mem-spark');
    if (svg) svg.querySelectorAll('.mem-spark-dot.active').forEach((d) => d.classList.remove('active'));
  },
};

// The spark's viewBox tracks its own pixel width (see render()), so a window
// resize needs a redraw to stay unstretched.
window.addEventListener('resize', () => {
  if (Mem.points.length) Mem.render();
});

// Hover anywhere over the spark: pick the nearest step by x rather than
// requiring a precise hit on a ~2px dot.
(function () {
  const svg = document.getElementById('mem-spark');
  if (!svg) return;
  svg.addEventListener('mousemove', (e) => {
    const i = Mem.nearestPoint(svg, e.clientX);
    if (i >= 0) Mem.showTooltip(i, e.clientX, e.clientY);
  });
  svg.addEventListener('mouseleave', Mem.hideTooltip);
})();
